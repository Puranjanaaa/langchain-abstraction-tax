# langchain-abstraction-tax

What does LangChain actually cost you, compared to leaner alternatives, for a
standard retrieval-augmented Q&A task? This repo builds the same RAG system
five times — once idiomatically in each of LangChain, LlamaIndex, and DSPy,
once as agentic OpenAI-SDK function-calling with no orchestration framework,
and once as a fully hand-rolled raw API baseline — over the same vendored
corpus, the same eval set, and byte-identical retrieval, so that whatever
differs in the results is attributable to the orchestration layer and
nothing else.

**The code here is evidence for a writeup, not the deliverable itself.** The
actual point is an honest, measured answer to "is the abstraction worth the
tax," not a working RAG app.

## How it's structured

```
corpus/kubernetes-docs/   396 vendored files (CC BY 4.0) from the Kubernetes
                           docs' Concepts + Tasks sections, pinned to one
                           commit so the corpus never drifts under the eval.
corpus/manifest.json      Per-file heading index (title, level, slug) — the
                           single source of truth for citation slugs.

eval/qa_set.jsonl         70 hand-verified Q&A pairs: 24 easy, 25 medium,
                           14 hard-multihop, 7 hard-unanswerable. Schema and
                           bucket definitions in eval/SCHEMA.md.
eval/harness.py           Loads qa_set.jsonl, runs an Implementation, times
                           each answer() call.
eval/metrics/             retrieval.py (recall@k, MRR), judge.py (LLM-judge
                           correctness + faithfulness), performance.py
                           (latency, token counts as a cost proxy).

shared/interface.py       The Implementation protocol every framework
                           satisfies: module-level `name` + `answer(question)`.
shared/types.py           Citation, Usage, Answer — the shared return shape.
shared/llm.py              Env-configured OpenAI-compatible client (see
                           Setup below) — every implementation's only path
                           to an LLM or embedding model.
shared/retrieval_config.py TOP_K, SECTION_CHAR_LIMIT, CHUNK_OVERLAP — locked
                           once, identical across all five implementations.
shared/chunking.py         Framework-agnostic header-aware chunking backing
                           that config — verified byte-identical to
                           LangChain's own splitter output across the whole
                           corpus, so no implementation's retrieval is
                           working from different chunk boundaries.

implementations/langchain_impl/         LCEL retriever | prompt | ChatOpenAI
implementations/llamaindex_impl/        VectorStoreIndex + retriever
implementations/dspy_impl/              Signature + ChainOfThought
implementations/function_calling_impl/  agentic search_docs tool-calling
implementations/raw_api_impl/           hand-rolled embed + cosine top-k,
                                         zero framework code
```

Every implementation embeds and retrieves from the exact same chunk
boundaries and top-k, and returns citations as `(file, heading)` pairs
resolvable against `corpus/manifest.json` — see each implementation's
module docstring for the specific methodology calls made for that
framework (e.g. why DSPy uses a typed `answerable` field instead of a
literal decline string, or why function-calling's tool loop distrusts
model content on tool-calling turns).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an OpenAI-compatible LLM +
embedding endpoint (developed against a local [LM Studio](https://lmstudio.ai)
server, but any compatible endpoint works).

```bash
uv sync
cp .env.example .env   # then fill in LLM_BASE_URL / LLM_API_KEY / LLM_MODEL / EMBEDDING_MODEL
```

## Quick start

`scripts/smoke_test.py` is the fastest way to see the whole project wired up
end to end — env config, a live chat + embedding call, the eval set loading,
and a real `answer()` call against every implementation:

```bash
uv run python scripts/smoke_test.py
```

For a closer look at one implementation — a small bucketed sample of the
eval set run through the full harness and metrics (retrieval recall/MRR,
LLM-judge correctness/faithfulness, latency, token cost) — see
`eval/harness.py` and `eval/metrics/`; each implementation was validated
this way while it was built.

Note: every implementation lazily builds its retrieval index (embedding all
~6.7k corpus chunks) on its first `answer()` call, so the first call in a
fresh process is much slower than the rest — expect a couple of minutes, not
seconds.

## Status

Phases 1–4 (corpus, shared scaffolding, LangChain implementation, eval
metrics) and Phase 5 (the remaining four implementations) are built and
individually validated against the harness. The full 70-question evaluation
run across all five implementations, and the tradeoff writeup itself, are
still ahead.
