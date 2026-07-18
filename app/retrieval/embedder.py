"""
app/retrieval/embedder.py
-------------------------
Factory for the shared embeddings client (Vertex AI backend, via
GoogleGenerativeAIEmbeddings from langchain-google-genai).

WHY THIS FILE EXISTS:
  Embeddings are the bridge between text and math. A question like
  "explain overfitting" becomes a 768-dimensional vector. Chunks from the
  PDF also become vectors. Retrieval is then similarity search in that
  vector space.

WHERE IT FITS:
  ingestion pipeline  -> get_embeddings() -> CognaraPGVectorStore.add_documents() (write path)
  ask_service         -> get_embeddings() -> CognaraPGVectorStore.similarity_search_with_score() (read path)

WHY GoogleGenerativeAIEmbeddings (not the older VertexAIEmbeddings):
  This module originally used langchain_google_vertexai.VertexAIEmbeddings,
  which was deprecated in langchain-google-vertexai 3.2.0 in favour of
  GoogleGenerativeAIEmbeddings from the newer langchain-google-genai package.
  This is NOT a switch to the consumer Gemini API — GoogleGenerativeAIEmbeddings
  supports the same Vertex AI backend (same GCP project, same billing, same
  text-embedding-004 model) via vertexai=True. We verified the class fields
  before switching rather than trusting the deprecation message blindly.

  Note: We did NOT make the equivalent migration for generation (ChatVertexAI
  -> ChatGoogleGenerativeAI) because that specific switch has reported 50-90%
  latency regressions due to REST vs gRPC. See generation.py.

IMPLEMENTATION NOTE:
  text-embedding-004 produces 768-dimensional vectors — matches
  settings.EMBEDDING_DIM and the chunks table's vector(768) column.

# Interview notes: local-notes/INTERVIEW_PREP.md — "app/retrieval/embedder.py"
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings

# Module-level singleton — safe to cache here because GoogleGenerativeAIEmbeddings
# uses plain sync HTTP under the hood (no async gRPC channel / event-loop binding
# risk). One instance is reused across all ingestion batches and /ask requests.
_embeddings_instance: GoogleGenerativeAIEmbeddings | None = None


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Return the shared embeddings instance, constructing it once on first call.

    vertexai=True + project=... selects the Vertex AI backend explicitly
    (billed to our GCP project, authenticated via Application Default
    Credentials). Without these, the class defaults to the consumer Gemini
    Developer API and looks for a GOOGLE_API_KEY instead.

    The same embeddings instance must be used for BOTH ingestion (embed chunks)
    and retrieval (embed queries) — a mismatch between models silently produces
    wrong similarity scores because vectors from different models live in
    incomparable spaces.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = GoogleGenerativeAIEmbeddings(
            model=settings.VERTEX_EMBEDDING_MODEL,
            project=settings.GCP_PROJECT_ID,
            location=settings.VERTEX_AI_LOCATION,
            vertexai=True,  # use Vertex AI backend, not consumer Gemini API
        )
    return _embeddings_instance
