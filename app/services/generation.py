"""
app/services/generation.py
---------------------------
Builds the evidence-grounded prompt and calls Gemini to generate an answer.

WHY THIS FILE EXISTS:
  Retrieval finds evidence; this file turns that evidence plus the user's
  question into a real, cited answer. It owns the prompt template and the
  Gemini call — nothing else in the codebase constructs a generation
  prompt or talks to the chat model directly.

WHERE IT FITS:
  ask_service.py -> retrieval (vector_store) -> generation (this file) -> AskResponse

WHY ChatVertexAI, NOT ChatGoogleGenerativeAI (a real, deliberate choice):
  Both VertexAIEmbeddings and ChatVertexAI raise the same
  LangChainDeprecationWarning, pointing at langchain_google_genai's
  replacement classes. For embeddings (see app/retrieval/embedder.py) we
  migrated, because the same-backend replacement had no reported
  downside. For the CHAT model specifically, multiple teams have reported
  50-90% latency increases after switching ChatVertexAI ->
  ChatGoogleGenerativeAI, attributed to the newer package using REST
  where the older one uses gRPC. Generation latency is directly
  user-facing (it's what makes /ask feel slow or fast), so this is
  exactly the class of regression worth avoiding. We keep the deprecated
  ChatVertexAI deliberately, with this reasoning documented, and will
  revisit once that regression is resolved upstream or measured
  acceptable against our own latency budget.

REAL BUG FOUND AND FIXED — NO MODULE-LEVEL SINGLETON FOR ASYNC gRPC CLIENTS:
  This file originally cached ChatVertexAI as a module-level singleton
  (the same pattern used successfully for embeddings and the vector
  store's SQLAlchemy engine). It broke, reproducibly: the SECOND async
  call to the shared instance in any given test session failed with
  "RuntimeError: Event loop is closed", even after ruling out sync/async
  mixing and asyncio.run() probe issues as the cause (three rounds of
  debugging — see the historical version of tests/integration/
  test_generation.py's docstring for the full trail).

  ROOT CAUSE, finally isolated: ChatVertexAI's async path uses grpc.aio,
  and a grpc.aio channel binds to whichever asyncio event loop is
  running when the channel is FIRST used. pytest-asyncio, by default
  (asyncio_default_test_loop_scope unset = function-scoped), creates a
  NEW event loop for every individual async test function and closes it
  when that test ends. So: test 1 creates the singleton, its gRPC
  channel binds to test 1's loop; test 1's loop closes; test 2 gets a
  fresh loop, but reuses the SAME singleton instance, whose channel is
  still bound to test 1's now-closed loop.

  This is not just a test artifact — it is a real constraint on any
  async gRPC client (which ChatVertexAI's ainvoke() is) reused across
  more than one event loop, which can also happen in production: a
  worker restart, a background task scheduled on a different loop, or
  any ASGI server that doesn't guarantee one single, permanent event
  loop for the process lifetime.

  FIX: do NOT cache ChatVertexAI as a module-level singleton. Construct
  a fresh instance per call to generate(). Constructing the client is a
  local object-setup cost (no network round trip), not a real
  performance concern, and it makes this code correct regardless of
  which event loop is calling it — the safer default for an async gRPC
  client. Contrast with embedder.py's GoogleGenerativeAIEmbeddings and
  vector_store.py's SQLAlchemy engine, both of which use plain sync
  HTTP/DB drivers under the hood and have no equivalent event-loop
  binding — singleton caching remains correct and worthwhile there.

PROMPT DESIGN — WHY CITATIONS ARE ENFORCED IN THE PROMPT, NOT JUST ASSUMED:
  Cognara Learn's whole value proposition is trustworthy, verifiable
  answers. An LLM asked a question with some context attached will often
  blend in outside knowledge it already has, not just what's in the
  provided evidence — this is a real, well-documented failure mode
  (unfaithfulness/hallucination even in RAG systems). The prompt
  explicitly instructs the model to answer ONLY from the provided
  evidence and to say so plainly if the evidence is insufficient, and
  every evidence chunk is given a citation TAG the model is told to
  reference. This doesn't guarantee faithfulness (verifying that is
  Layer 4's evidence-gate/Layer 9's eval job, not this file's), but it
  is the first, cheapest line of defence.

INTERVIEW EXPLANATION:
  "The generation module has exactly one job: take retrieved evidence and
  a question, and produce a grounded answer. The prompt explicitly tells
  Gemini to answer only from the provided chunks and to say when the
  evidence is insufficient — that's a deliberate anti-hallucination
  measure. I also learned, the hard way, that async gRPC clients like
  ChatVertexAI shouldn't be cached as module-level singletons — their
  channel binds to whichever event loop is active on first use, so
  reusing one instance across a NEW event loop later fails with 'Event
  loop is closed'. I construct a fresh client per call instead, which
  costs a small amount of local object setup but is correct regardless
  of which event loop calls it — important for anything beyond a single,
  permanently-running server loop."
"""

from langchain_core.documents import Document
from langchain_google_vertexai import ChatVertexAI

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are Cognara Learn, an evidence-verified AI learning copilot.

You answer ONLY using the numbered evidence chunks provided below. Follow these rules strictly:

1. Base your answer entirely on the provided evidence. Do not use outside knowledge, even if you know the answer.
2. When you use a fact from a chunk, reference it by its tag, like [1] or [2].
3. If the evidence does not contain enough information to answer the question, say so plainly instead of guessing.
4. Write in simple, clear English. Explain technical terms the first time you use them.
5. Keep the answer focused and complete, but do not pad it with unnecessary text.
"""


def _get_llm() -> ChatVertexAI:
    """
    Construct a fresh ChatVertexAI instance per call — deliberately NOT
    a cached module-level singleton. See the module docstring's REAL BUG
    FOUND AND FIXED section for why: its async gRPC channel binds to
    whichever event loop is running at first use, and reusing a cached
    instance across a different/later event loop fails with
    "RuntimeError: Event loop is closed".
    """
    return ChatVertexAI(
        model_name=settings.VERTEX_GENERATION_MODEL,
        project=settings.GCP_PROJECT_ID,
        location=settings.VERTEX_AI_LOCATION,
        temperature=0.2,  # low temperature: favour grounded, consistent answers over creative ones
        max_output_tokens=1024,
    )


def _build_evidence_block(chunks: list[Document]) -> str:
    """
    Format retrieved chunks into a numbered evidence block the prompt can
    reference by tag, e.g. [1], [2]. Each tag also carries its citation
    so the model can be consistent about what [1] refers to.
    """
    lines = []
    for i, doc in enumerate(chunks, start=1):
        meta = doc.metadata
        source = f"{meta.get('course_name', 'unknown')} — {meta.get('chapter', 'unknown')}"
        if meta.get("topic"):
            source += f" — {meta['topic']}"
        page = meta.get("page_range") or meta.get("page_number")
        lines.append(f"[{i}] (Source: {source}, page {page})\n{doc.page_content}")
    return "\n\n".join(lines)


async def generate(question: str, chunks: list[Document]) -> tuple[str, int]:
    """
    Generate an evidence-grounded answer to `question` using `chunks` as
    the only permitted source of facts.

    Args:
        question: The user's question, verbatim.
        chunks: Retrieved evidence Documents (from
            CognaraPGVectorStore.similarity_search_with_score), already
            filtered/thresholded by the caller — this function does not
            re-check relevance, it trusts what it's given.

    Returns:
        (answer_text, total_tokens_used) — total_tokens_used is the real
        figure from Gemini's response metadata, not an estimate.
    """
    llm = _get_llm()
    evidence_block = _build_evidence_block(chunks)

    user_prompt = (
        f"Evidence:\n{evidence_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer the question using only the evidence above, citing chunk "
        f"numbers like [1] where relevant."
    )

    logger.info("generation_start", chunk_count=len(chunks))

    response = await llm.ainvoke([
        ("system", SYSTEM_PROMPT),
        ("human", user_prompt),
    ])

    total_tokens = 0
    if response.usage_metadata:
        total_tokens = response.usage_metadata.get("total_tokens", 0)

    logger.info("generation_done", tokens_used=total_tokens, answer_chars=len(response.content))

    return response.content, total_tokens
