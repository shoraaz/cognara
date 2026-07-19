"""
app/services/quiz_interview.py
---------------------------------
Layer 5 (part 2): Quiz and Interview — the two stateful learning modes.
See ADR 0009 for the full schema and design reasoning.

WHY ONE MODULE FOR BOTH MODES:
  Quiz and Interview share the identical session/turn persistence layer
  (learning_sessions, learning_session_turns — see ADR 0009) and the
  identical CRAG-retrieve -> Gemini-writes-question -> user-answers ->
  Gemini-grades-answer flow. The only real difference is HOW the next
  question is chosen: Quiz optimizes for topic coverage (don't repeat a
  sub-topic already asked), Interview adapts difficulty based on the
  previous answer's correctness (dig deeper or step back). That
  difference lives entirely in _choose_focus_hint() below — everything
  else is shared code, not duplicated per mode.

WHY THE ANSWER KEY IS GENERATED BUT NEVER SHOWN TO THE USER:
  When a question is created, Gemini also writes an internal answer_key_text
  grounded in the same evidence — this is stored in learning_session_turns
  but never included in any response the user sees. Grading the user's
  actual answer compares it against BOTH the answer key AND the original
  evidence (not just the key alone), the same "compare against real
  evidence" principle Layer 4's faithfulness check uses — a canned answer
  key can itself drift from the evidence over a long question, so grading
  re-grounds against the source, not just the key.

WHY SESSION STATE LIVES IN CLOUD SQL, NOT ADK's InMemorySessionService:
  CRAG's session pattern is explicitly one-shot, discarded after a single
  request (ADR 0006). A quiz session must survive across multiple separate
  HTTP requests — the user answers question 1, then later sends question
  2's answer in a NEW request. That requires real persistence, not an
  in-memory object scoped to one Runner.run_async() call.

REAL BUG FOUND AND FIXED (round 1) — DOUBLE-DESERIALIZING A jsonb COLUMN:
  The first real end-to-end quiz session (create -> Q1 -> submit answer)
  crashed on submit_answer() with: "TypeError: the JSON object must be
  str, bytes or bytearray, not list". Root cause: evidence_chunk_ids is
  stored as a jsonb column, and pg8000/SQLAlchemy ALREADY deserializes a
  jsonb value back into a native Python list automatically when the row
  is read — calling json.loads() on it again fails, since json.loads()
  only accepts str/bytes, not an already-parsed list.
  FIX: read evidence_chunk_ids directly as a Python list, no json.loads()
  needed on the read path. json.dumps() on the WRITE path is still
  correct/necessary — the asymmetry was only caught by running the full
  round trip.

REAL BUG FOUND AND FIXED (round 2) — THE "AVOID REPEATING" HINT NEVER
ACTUALLY REACHED THE QUESTION WRITER:
  A real two-question quiz session produced near-identical Q1/Q2 even
  though CRAG's retrieval genuinely found DIFFERENT evidence for Q2 —
  the focus_hint text steered the CRAG RETRIEVAL query (which worked),
  but was NEVER passed into the question-WRITER's own prompt, so the
  writer had no visibility into what had already been asked.
  FIX: _next_turn() now passes prior question text directly into the
  writer's prompt via _build_writer_context(), not just the retrieval
  query.

REAL BUG FOUND AND FIXED (round 3) — "STEP BACK" WAS AMBIGUOUS ENOUGH
THAT CRAG LEFT THE TOPIC ENTIRELY:
  A real Interview-mode session: after an answer graded incorrect, the
  hint told CRAG to ask "a more foundational question, stepping back
  from: [the vanishing gradient question]". CRAG's own critic read this
  LITERALLY — it graded the (correct, on-topic) vanishing-gradient
  evidence as low-relevance specifically BECAUSE it was about vanishing
  gradients, reasoning that the user had "explicitly asked to step away
  from" that subject. Its own rewrite_query then searched for vanishing
  gradients OUTSIDE neural networks entirely — which does not exist in
  this corpus — and correctly abstained rather than force a bad answer.
  The session-level result: submit_answer() returned next_question=None,
  a real functional dead end, even though the underlying topic had
  plenty of usable evidence at a simpler level.
  FIX: reworded the hint to be explicit that "foundational" means
  SIMPLER content on the SAME topic, never a different subject — "ask
  about a more basic, introductory aspect of the SAME topic ({topic}),
  not a different subject." Also added a safety net: if a stateful-mode
  turn's CRAG call abstains, retry ONCE with the hint removed (plain
  topic only) before giving up — since the topic itself is known-good
  (it was validated to have evidence at session start), a hint-caused
  abstain should not be allowed to dead-end an otherwise healthy session.

INTERVIEW EXPLANATION:
  "I hit three real bugs building this, and the third is the most
  interesting: I told CRAG's critic to help the question 'step back' to
  something simpler, and it interpreted that literally as 'actively
  avoid the original subject,' searched for vanishing gradients outside
  neural networks — which doesn't exist in my corpus — and correctly,
  honestly abstained rather than force a bad answer. The retrieval
  system did exactly what I told it to; I just told it something
  ambiguous. I fixed the wording to be explicit that 'foundational'
  means simpler content on the SAME topic, and added a safety net that
  retries without the steering hint at all if a hinted search abstains,
  since the underlying topic was already confirmed to have real
  evidence when the session started."
"""

import uuid
from typing import Literal

import sqlalchemy
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field

from app.agents.crag_runner import run_crag
from app.core.config import settings
from app.core.logging import get_logger
from ingestion.pipelines.init_db import get_engine

logger = get_logger(__name__)

SessionMode = Literal["quiz", "interview"]


# ── Structured outputs for question-writing and answer-grading ──────────────

class GeneratedQuestion(BaseModel):
    question_text: str = Field(..., description="The question to ask the user.")
    answer_key_text: str = Field(..., description="The correct answer, grounded in the evidence. Never shown to the user directly.")


class GradedAnswer(BaseModel):
    is_correct: bool = Field(..., description="True if the user's answer is substantively correct, judged against the evidence and answer key.")
    feedback_text: str = Field(..., description="Short, encouraging feedback explaining what was right or wrong, citing the evidence.")


QUESTION_WRITER_PROMPT_QUIZ = """You are Cognara Learn, writing ONE quiz question grounded strictly \
in the evidence below.

1. Write a single, clear question testing understanding of a concept in the evidence.
2. If a list of ALREADY-ASKED questions is provided, your new question MUST test a substantively \
DIFFERENT fact, mechanism, or angle than every one of them — not just different wording of the \
same underlying question. Read them carefully before writing.
3. Also write the correct answer_key_text, grounded entirely in the evidence — this is for \
internal grading only and will not be shown to the user.
4. Do not invent facts not present in the evidence.
"""

QUESTION_WRITER_PROMPT_INTERVIEW = """You are Cognara Learn, acting as a technical interviewer, \
writing ONE follow-up question grounded strictly in the evidence below.

1. If told the PREVIOUS answer was CORRECT, write a HARDER, more probing question that digs \
deeper into the same sub-topic than the previous question did — the way a real interviewer \
follows up on a strong answer. It must go MEANINGFULLY deeper, not just reword the same question.
2. If told the PREVIOUS answer was INCORRECT, or this is the first question, write a more \
FOUNDATIONAL question — a SIMPLER angle of the SAME topic, or a prerequisite concept the evidence \
covers. Do NOT change the subject matter entirely — stay on the same overall topic, just simpler.
3. Also write the correct answer_key_text, grounded entirely in the evidence — internal use only.
4. Do not invent facts not present in the evidence.
"""

ANSWER_GRADER_PROMPT = """You are grading a user's answer to a learning question. You will be given \
the original evidence, the question, the internal answer key, and the user's actual answer.

Judge is_correct based on whether the user's answer is SUBSTANTIVELY correct according to the \
EVIDENCE (not just whether it matches the answer key's wording) — a differently-worded but \
factually correct answer is still correct. Write short, encouraging feedback_text explaining \
what was right or what was missed, citing the evidence.
"""


def _get_structured_llm(schema: type[BaseModel], temperature: float = 0.3):
    """Fresh ChatVertexAI + with_structured_output() per call — not a
    cached singleton. See faithfulness.py / generation.py's documented
    lesson about async gRPC clients and event loops."""
    llm = ChatVertexAI(
        model_name=settings.VERTEX_GENERATION_MODEL,
        project=settings.GCP_PROJECT_ID,
        location=settings.VERTEX_AI_LOCATION,
        temperature=temperature,
    )
    return llm.with_structured_output(schema)


# ── Persistence ───────────────────────────────────────────────────────────────

def _get_db_engine():
    return get_engine(ip_type="PUBLIC")


def create_session(mode: SessionMode, topic: str, course_filter: str | None = None) -> str:
    """Create a new learning session row. Returns the new session_id."""
    session_id = uuid.uuid4().hex
    engine = _get_db_engine()
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text("""
            INSERT INTO learning_sessions (session_id, mode, topic, course_filter)
            VALUES (:session_id, :mode, :topic, :course_filter);
        """), {"session_id": session_id, "mode": mode, "topic": topic, "course_filter": course_filter})
        conn.commit()
    logger.info("session_created", session_id=session_id, mode=mode, topic=topic)
    return session_id


def _get_session(session_id: str) -> dict | None:
    engine = _get_db_engine()
    with engine.connect() as conn:
        row = conn.execute(sqlalchemy.text(
            "SELECT session_id, mode, topic, course_filter, status FROM learning_sessions WHERE session_id = :id;"
        ), {"id": session_id}).mappings().fetchone()
    return dict(row) if row else None


def _get_turns(session_id: str) -> list[dict]:
    engine = _get_db_engine()
    with engine.connect() as conn:
        rows = conn.execute(sqlalchemy.text("""
            SELECT turn_id, turn_number, question_text, answer_key_text, user_answer_text,
                   is_correct, feedback_text, evidence_chunk_ids
            FROM learning_session_turns WHERE session_id = :id ORDER BY turn_number;
        """), {"id": session_id}).mappings().fetchall()
    return [dict(r) for r in rows]


def _save_turn(session_id: str, turn_number: int, question: GeneratedQuestion, evidence_chunk_ids: list[str]) -> str:
    import json
    turn_id = uuid.uuid4().hex
    engine = _get_db_engine()
    with engine.connect() as conn:
        # WRITE path: json.dumps() IS correct/necessary here — the pg8000
        # jsonb bind parameter expects a JSON-encoded string to send to
        # Postgres, which stores it as jsonb. See module docstring for why
        # this is NOT symmetric with the read path below.
        conn.execute(sqlalchemy.text("""
            INSERT INTO learning_session_turns
                (turn_id, session_id, turn_number, question_text, answer_key_text, evidence_chunk_ids)
            VALUES (:turn_id, :session_id, :turn_number, :question_text, :answer_key_text, :evidence_chunk_ids);
        """), {
            "turn_id": turn_id, "session_id": session_id, "turn_number": turn_number,
            "question_text": question.question_text, "answer_key_text": question.answer_key_text,
            "evidence_chunk_ids": json.dumps(evidence_chunk_ids),
        })
        conn.commit()
    return turn_id


def _save_grade(turn_id: str, user_answer_text: str, graded: GradedAnswer) -> None:
    engine = _get_db_engine()
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text("""
            UPDATE learning_session_turns
            SET user_answer_text = :answer, is_correct = :correct, feedback_text = :feedback
            WHERE turn_id = :turn_id;
        """), {
            "turn_id": turn_id, "answer": user_answer_text,
            "correct": graded.is_correct, "feedback": graded.feedback_text,
        })
        conn.commit()


# ── Core flow ─────────────────────────────────────────────────────────────────

async def start_session(mode: SessionMode, topic: str, course_filter: str | None = None) -> dict:
    """
    Start a new Quiz or Interview session: create the session row, run
    CRAG for the first question's evidence, write question 1.

    Returns: {session_id, turn_id, question_text, abstained, abstain_reason}
    """
    session_id = create_session(mode, topic, course_filter)
    return await _next_turn(session_id, mode, topic, course_filter, turn_history=[])


async def submit_answer(session_id: str, user_answer_text: str) -> dict:
    """
    Grade the user's answer to the CURRENT (most recent, ungraded) turn,
    then generate the NEXT question.

    Returns: {is_correct, feedback_text, next_question: {...} | None}
    """
    session = _get_session(session_id)
    if session is None:
        raise ValueError(f"No session found for session_id={session_id}")

    turns = _get_turns(session_id)
    if not turns:
        raise ValueError(f"Session {session_id} has no turns to answer")

    current_turn = turns[-1]
    if current_turn["user_answer_text"] is not None:
        raise ValueError(f"Turn {current_turn['turn_number']} in session {session_id} was already answered")

    # READ path: evidence_chunk_ids comes back from pg8000/SQLAlchemy ALREADY
    # deserialized into a native Python list (jsonb columns auto-decode on
    # read) — NOT a JSON string needing json.loads(). See module docstring's
    # REAL BUG FOUND AND FIXED (round 1) for the exact crash this caused.
    chunk_ids = current_turn["evidence_chunk_ids"]
    evidence_text = _fetch_chunk_texts(chunk_ids)

    grader = _get_structured_llm(GradedAnswer, temperature=0.0)
    grading_prompt = (
        f"Evidence:\n{evidence_text}\n\n"
        f"Question: {current_turn['question_text']}\n\n"
        f"Answer key (internal): {current_turn['answer_key_text']}\n\n"
        f"User's answer: {user_answer_text}"
    )
    graded = await grader.ainvoke([("system", ANSWER_GRADER_PROMPT), ("human", grading_prompt)])

    _save_grade(current_turn["turn_id"], user_answer_text, graded)
    logger.info("answer_graded", session_id=session_id, turn_number=current_turn["turn_number"], is_correct=graded.is_correct)

    # Refresh turn history (now includes the just-graded answer) for choosing the next question.
    turns = _get_turns(session_id)
    next_result = await _next_turn(
        session_id, session["mode"], session["topic"], session["course_filter"], turn_history=turns,
    )

    return {
        "is_correct": graded.is_correct,
        "feedback_text": graded.feedback_text,
        "next_question": next_result,
    }


def _fetch_chunk_texts(chunk_ids: list[str]) -> str:
    if not chunk_ids:
        return ""
    engine = _get_db_engine()
    with engine.connect() as conn:
        rows = conn.execute(sqlalchemy.text(
            "SELECT text FROM chunks WHERE chunk_id = ANY(:ids);"
        ), {"ids": chunk_ids}).fetchall()
    return "\n\n".join(r[0] for r in rows)


def _choose_focus_hint(mode: SessionMode, topic: str, turn_history: list[dict]) -> str:
    """
    Build a short natural-language hint appended to the CRAG RETRIEVAL
    query, steering retrieval toward the right next sub-topic. This is
    where Quiz and Interview genuinely diverge — see module docstring.

    IMPORTANT (see module docstring, REAL BUG FOUND AND FIXED round 3):
    the hint must stay unambiguously ANCHORED to the original topic —
    "foundational"/"simpler" must never read as "a different subject,"
    or CRAG's own critic can (and did, in a real run) interpret it as an
    instruction to avoid the topic entirely, search for a foundational
    concept that doesn't exist in this corpus, and abstain.
    """
    if not turn_history:
        return ""

    if mode == "quiz":
        covered = [t["question_text"] for t in turn_history]
        return f" (still about {topic} — already asked about: " + "; ".join(covered[-3:]) + " — cover a different angle of the SAME topic)"

    # interview mode: adapt on the MOST RECENT answered turn's correctness
    last = turn_history[-1]
    if last["is_correct"] is True:
        return f" (still about {topic} — previous answer was correct, ask a harder, deeper follow-up on the SAME topic)"
    elif last["is_correct"] is False:
        return (
            f" (still about {topic} — previous answer was incorrect, ask about a more basic, "
            f"introductory aspect of the SAME topic ({topic}), NOT a different subject)"
        )
    return ""


def _build_writer_context(mode: SessionMode, turn_history: list[dict]) -> str:
    """
    Build the context appended to the QUESTION-WRITER's own prompt (not
    just the retrieval query — see module docstring, REAL BUG FOUND AND
    FIXED round 2). This is what actually lets the writer avoid
    repeating a prior question, since it now SEES the prior questions
    directly, not just evidence that happens to have been retrieved
    with repetition-avoidance in mind.
    """
    if not turn_history:
        return ""

    if mode == "quiz":
        already_asked = "\n".join(f"- {t['question_text']}" for t in turn_history)
        return f"\n\nQuestions already asked in this quiz (write something substantively different):\n{already_asked}"

    last = turn_history[-1]
    correctness = "CORRECT" if last["is_correct"] else "INCORRECT"
    return (
        f"\n\nPrevious question: {last['question_text']}\n"
        f"The user's answer to it was: {correctness}\n"
        f"Write your next question accordingly (see system instructions)."
    )


async def _next_turn(
    session_id: str, mode: SessionMode, topic: str, course_filter: str | None, turn_history: list[dict],
) -> dict:
    """
    Shared: run CRAG for the next question's evidence, write the
    question, save the turn. Includes a safety net (see module
    docstring, REAL BUG FOUND AND FIXED round 3): if the HINTED search
    abstains, retry once with the plain topic and no hint before giving
    up — the topic itself is known-good (validated at session start),
    so a hint-caused abstain should not dead-end an otherwise healthy
    session.
    """
    focus_hint = _choose_focus_hint(mode, topic, turn_history)
    crag_result = await run_crag(topic + focus_hint, course_filter=course_filter)
    grade = crag_result["grade"]
    evidence_chunks = crag_result["evidence_chunks"]

    if (grade["decision"] == "abstain" or not evidence_chunks) and focus_hint:
        logger.info("next_turn_hint_caused_abstain_retrying_plain_topic", session_id=session_id, topic=topic)
        crag_result = await run_crag(topic, course_filter=course_filter)
        grade = crag_result["grade"]
        evidence_chunks = crag_result["evidence_chunks"]

    if grade["decision"] == "abstain" or not evidence_chunks:
        return {
            "session_id": session_id, "turn_id": None, "question_text": None,
            "abstained": True, "abstain_reason": grade["reason"],
        }

    evidence_text = "\n\n".join(c["text"] for c in evidence_chunks)
    writer_context = _build_writer_context(mode, turn_history)
    writer_prompt = QUESTION_WRITER_PROMPT_QUIZ if mode == "quiz" else QUESTION_WRITER_PROMPT_INTERVIEW
    writer = _get_structured_llm(GeneratedQuestion, temperature=0.4)
    question = await writer.ainvoke([
        ("system", writer_prompt),
        ("human", f"Evidence:\n{evidence_text}\n\nTopic: {topic}{writer_context}"),
    ])

    turn_number = len(turn_history) + 1
    chunk_ids = [c["chunk_id"] for c in evidence_chunks if c.get("chunk_id")]
    turn_id = _save_turn(session_id, turn_number, question, chunk_ids)

    logger.info("question_generated", session_id=session_id, mode=mode, turn_number=turn_number)

    return {
        "session_id": session_id, "turn_id": turn_id,
        "question_text": question.question_text,
        "turn_number": turn_number, "abstained": False, "abstain_reason": None,
    }
