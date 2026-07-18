"""
app/models/schemas.py
---------------------
Pydantic models for all API request and response bodies.

WHY THIS FILE EXISTS:
  Every boundary between modules (API → service → retrieval) should have
  a typed contract. Pydantic validates at runtime AND generates OpenAPI docs
  automatically. If a field is missing or the wrong type, we get a clear
  422 error, not a silent KeyError deep in the stack.

# Interview notes: local-notes/INTERVIEW_PREP.md — "app/models/schemas.py"
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Request ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    course_filter: Optional[str] = Field(
        default=None,
        description="Restrict retrieval to one course. "
                    "Example: '100 Days of Machine Learning'",
    )
    chapter_filter: Optional[str] = Field(
        default=None,
        description="Further restrict to one chapter within the course.",
    )


class LearningModeRequest(BaseModel):
    """
    Request shape for Layer 5's single-turn learning modes (Explain,
    Compare, Study-Plan) — see ADR 0008. topic_b is only used by Compare
    mode, and is REQUIRED for it (validated in learning_modes.py, not
    here, since the requirement is mode-conditional on which endpoint/
    mode value is used, not a fixed schema rule).
    """
    topic: str = Field(..., min_length=3, max_length=500, description="The main topic or question.")
    topic_b: Optional[str] = Field(
        default=None,
        description="The second topic to compare against — REQUIRED for Compare mode only.",
    )
    course_filter: Optional[str] = None
    chapter_filter: Optional[str] = None


# ── Citation ──────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    course_name: str
    chapter: str
    topic: Optional[str] = None
    page_number: int
    page_range: Optional[str] = None
    relevance_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Similarity score from vector retrieval (0–1).",
    )


# ── Response ──────────────────────────────────────────────────────────────────

class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: str = Field(
        ...,
        description="'high' | 'medium' | 'low' | 'abstained'",
    )
    abstained: bool = Field(
        default=False,
        description="True when the system refuses to answer due to "
                    "insufficient evidence.",
    )
    abstain_reason: Optional[str] = Field(
        default=None,
        description="Reason for abstention, shown to the user.",
    )
    was_regenerated: bool = Field(
        default=False,
        description="True when Layer 4's faithfulness check flagged the "
                    "first generated answer as unsupported by its evidence, "
                    "and this is the corrected, regenerated version. See "
                    "ADR 0007.",
    )
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
