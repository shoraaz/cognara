"""
app/retrieval/embedder.py
-------------------------
Factory for the shared embeddings client (Vertex AI backend, via
GoogleGenerativeAIEmbeddings).

WHY THIS FILE EXISTS:
  Embeddings are the bridge between text and math. A question like
  "explain overfitting" becomes a 768-dimensional vector. Chunks from the
  PDF also become vectors. Retrieval is then similarity search in that
  vector space.

REFACTOR NOTE (see ADR 0004, and a real deprecation caught mid-Module-4):
  This module originally used langchain_google_vertexai.VertexAIEmbeddings.
  Constructing it worked, but raised a real LangChainDeprecationWarning:
  that class is deprecated as of langchain-google-vertexai 3.2.0, in
  favour of GoogleGenerativeAIEmbeddings from the newer, consolidated
  langchain-google-genai package (4.0.0+).

  This is NOT a switch to the consumer Gemini API. GoogleGenerativeAIEmbeddings
  supports the SAME Vertex AI backend — same GCP project, same billing,
  same text-embedding-004 model — selected explicitly with vertexai=True
  and project=<our project>. We verified this by checking the class's
  real fields (model, project, location, vertexai all present) before
  switching, not by assuming the deprecation warning's suggested
  replacement was a drop-in without checking.

  We did NOT make the equivalent switch for generation (ChatVertexAI is
  also deprecated in favour of ChatGoogleGenerativeAI) — several teams
  have reported 50-90% latency increases after that specific migration,
  attributed to the new package using REST where the old one used gRPC.
  Embeddings showed no equivalent reported issue, so we migrated that
  one now and left ChatVertexAI in place for Module 6 (generation),
  revisiting once the latency concern is resolved upstream or measured
  directly against our own workload.

WHERE IT FITS:
  ingestion pipeline  → get_embeddings() → CognaraPGVectorStore.add_documents() (write path)
  ask_service         → get_embeddings() → CognaraPGVectorStore.similarity_search_with_score() (read path)

IMPLEMENTATION NOTE:
  Uses text-embedding-004 (settings.VERTEX_EMBEDDING_MODEL) via the
  Vertex AI backend. This model is chosen because:
  - 768 dimensions (good quality/cost balance) — matches
    settings.EMBEDDING_DIM and the chunks table's vector(768) column
  - Available on GCP with our project's credentials
  - Batch-friendly for ingestion
  We do NOT use OpenAI embeddings — our GCP-first commitment means we
  avoid multi-cloud dependencies at this stage (see ADR 0002).

INTERVIEW EXPLANATION:
  "The embedder is stateless — give it text, get a vector back. I use
  LangChain's GoogleGenerativeAIEmbeddings, explicitly configured with
  vertexai=True and our GCP project, so it calls Vertex AI's
  text-embedding-004, not the consumer Gemini API — same backend as
  before, just through the actively-maintained package instead of a
  deprecated one. I checked the actual class fields before switching
  rather than trusting the deprecation warning's suggestion blindly,
  and I deliberately did NOT make the equivalent switch for the chat
  model, because that migration has real, currently-reported latency
  regressions elsewhere. I'd rather carry one documented deprecation
  warning than introduce a measured performance regression."
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings

_embeddings_instance: GoogleGenerativeAIEmbeddings | None = None


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Return the shared embeddings instance, constructing it once
    (module-level singleton) so repeated calls across a single process —
    e.g. many ingestion batches, or many /ask requests — reuse the same
    underlying client rather than re-authenticating every time.

    vertexai=True + project=... selects the Vertex AI backend explicitly
    (billed to our GCP project, uses Application Default Credentials) —
    without it, this class would default to the consumer Gemini
    Developer API and look for a GOOGLE_API_KEY instead.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = GoogleGenerativeAIEmbeddings(
            model=settings.VERTEX_EMBEDDING_MODEL,
            project=settings.GCP_PROJECT_ID,
            location=settings.VERTEX_AI_LOCATION,
            vertexai=True,
        )
    return _embeddings_instance
