# ADR 0004: Ingestion and Orchestration Framework Selection — LangChain + ADK, No LangGraph

Status: Accepted
Date: 2026-07-17

## Context

Modules 1-2 (PDF parser, chunker) were originally built as plain Python
functions with no framework — free functions in `ingestion/parsers/` and
`ingestion/chunking/`, tested directly against the real corpus PDFs.

Two questions came up mid-Phase-1 that this ADR settles:

1. Should ingestion (parsing, chunking, vector storage) use a framework
   like LangChain, or stay as hand-rolled functions? KodeMellow (a related
   prior project) used a LangChain docs adapter with an Ingestion Adapter
   ABC pattern — Cognara's plain-function approach diverged from that
   precedent without an explicit decision being made at the time.
2. Which framework should own agent/orchestration behaviour for Layers
   3-9 (CRAG, evidence gating, learning modes, GraphRAG, agents/MCP/A2A,
   guardrails)? KodeMellow used Google's Agent Development Kit (ADK) on
   Vertex AI. The candidates for Cognara were ADK, LangGraph, or a mix.

## Decision

**LangChain** is used as the *component library* for ingestion and
retrieval (Layers 1-2):
- Document loaders (custom `BaseLoader` subclass wrapping PyMuPDF, since
  no stock LangChain loader supports our page-range-restricted extraction)
- Text splitters (custom `TextSplitter` subclass carrying our
  heading-aware chunking logic, since no stock splitter knows our
  corpus's LaTeX-numbered-heading structure)
- `PGVector` vectorstore wrapper over our Cloud SQL + pgvector instance
- `VertexAIEmbeddings` and `ChatVertexAI` for embedding and generation

**ADK (Agent Development Kit)** owns all stateful orchestration and agent
behaviour from Layer 3 onward:
- L3 CRAG: a single ADK Agent with tools (`search_notes`,
  `grade_retrieval`, `rewrite_query`); ADK's own reasoning loop drives the
  retrieve -> grade -> retry decision, rather than a hand-built graph
- L4 Self-RAG / evidence gate: an extension of the same agent's tool-set
- L5 Learning modes (Explain/Compare/Quiz/Interview/Study-Plan): five ADK
  sub-agents or tools under one orchestrator, matching KodeMellow's
  multi-agent pattern
- L6 GraphRAG: Neo4j access via LangChain's `Neo4jGraph` under the hood,
  exposed to the agent layer as an ADK tool (`query_concept_graph`)
- L7 Agents/MCP/A2A: ADK orchestrator + specialized agents + MCP tool
  definitions + A2A, as originally planned
- L8 Guardrails: ADK callbacks for pre/post tool and agent-turn checks,
  plus LangChain output parsers for validating structured output shape

**LangGraph is explicitly rejected** for this project.

## Alternatives considered

| Option | Learning curve | GCP/deploy fit | Ecosystem maturity | Decision |
|---|---|---|---|---|
| **ADK** | Lower — agents are plain Python classes/functions with built-in orchestrators (SequentialAgent, ParallelAgent, LoopAgent); you declare flow, not graph mechanics | Strong — `adk deploy` to Cloud Run, `adk eval` for regression testing, native Vertex AI integration | Smaller than LangChain's, but includes native first-class MCP AND A2A support (the only framework of its peers with both) | **CHOSEN** for L3-L9 |
| **LangGraph** | Higher — requires modelling agents as explicit directed graphs: typed state, nodes, conditional edges, checkpointers | Model-agnostic, deploy-anywhere, but no special GCP integration beyond what LangChain already offers | Most mature agent-orchestration ecosystem in 2026; best-in-class observability via LangSmith | REJECTED: steeper learning curve for the same reasoning ADK gives more simply, and introduces a second orchestration paradigm alongside ADK's KodeMellow precedent rather than one consistent mental model |
| **Plain functions (no framework), continued** | Lowest to write, but no reusable abstractions, no interoperability with MCP/A2A tooling, harder to extend into a multi-agent system later | N/A | N/A | REJECTED: Layer 7 already commits to MCP tools and A2A; building that on raw functions means reinventing what ADK already provides |

## Why ADK over LangGraph specifically

1. **Lower learning curve.** ADK expresses orchestration as ordinary
   Python control flow with named orchestrator types; LangGraph requires
   thinking in graph theory (nodes, typed state schemas, conditional
   edges) before writing the first line of actual logic.
2. **GCP-native deployment story.** `adk deploy` targets Cloud Run
   directly; `adk eval` gives built-in regression testing. Replicating
   this in the LangGraph ecosystem takes real additional work.
3. **First-class MCP and A2A support.** ADK is the only framework in its
   comparison class with native support for both protocols — MCP for
   tool integration, A2A for cross-agent communication — which Layer 7
   already required.
4. **Consistency with the KodeMellow precedent.** One agent-orchestration
   mental model across the portfolio, rather than ADK on one project and
   LangGraph on another for the same category of problem.

## Trade-off acknowledged

LangGraph has a more mature third-party ecosystem and the strongest
observability story in the market via LangSmith. ADK is tightly coupled
to Gemini and Google Cloud, which is a real constraint for a
multi-cloud or multi-model project. For Cognara Learn — a GCP-committed,
interview-preparation-focused build where "explain this trade-off simply"
is itself a project goal — that coupling is treated as a feature, not a
cost.

## Consequences

- Modules 1 and 2 (`pdf_parser.py`, `chunker.py`) are refactored from
  plain functions into a LangChain `BaseLoader` subclass and a LangChain
  `TextSplitter` subclass respectively. Existing test suites are
  preserved and re-verified against the real corpus after the refactor,
  not replaced blindly — the heading-detection and page-numbering logic
  proven in Modules 1-2 does not change, only the interface it is exposed
  through.
- `vector_store.py`'s hand-rolled SQL is replaced by LangChain's
  `PGVector` wrapper over the same Cloud SQL + pgvector instance
  (ADR 0003 is unaffected — the storage engine choice does not change,
  only the client library used to talk to it).
- `langchain`, `langchain-google-vertexai`, `langchain-postgres`, and
  `google-adk` are added to `pyproject.toml`.
- No `langgraph` dependency is added. If a future requirement genuinely
  needs LangGraph-style explicit graph control that ADK cannot express
  cleanly, that would be a new ADR with the specific gap named, not a
  silent reintroduction.

## Interview summary

"I use LangChain for the ingestion and retrieval components — loaders,
splitters, the pgvector vectorstore wrapper, embeddings and generation
clients — because those map directly onto LangChain's component
abstractions. For orchestration — the CRAG retry loop, the evidence gate,
the five learning modes, and the agent/MCP/A2A layer — I use Google's ADK
instead of LangGraph. ADK expresses that logic as agents and tools in
plain Python, which was a faster learning curve for me than modelling the
same behaviour as an explicit LangGraph state graph, and it deploys
natively to Cloud Run and has first-class support for both MCP and A2A,
which I already needed for the agent layer. I'm aware LangGraph has a
more mature ecosystem and better third-party observability — for a
GCP-committed project where I also wanted the simplest correct mental
model, ADK's tighter Google Cloud integration was the right trade to
make."
