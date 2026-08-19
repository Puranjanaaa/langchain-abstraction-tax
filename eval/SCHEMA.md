# Eval set schema

`eval/qa_set.jsonl` — one JSON object per line.

```json
{
  "id": "k8s-001",
  "question": "string",
  "bucket": "easy | medium | hard-multihop | hard-unanswerable",
  "reference_answer": "string",
  "citations": [
    {"file": "concepts/workloads/controllers/deployment.md", "heading": "rolling-back-a-deployment"}
  ],
  "unanswerable": false,
  "notes": "why this question belongs in this bucket"
}
```

- `citations[].file` is the path relative to `corpus/kubernetes-docs/`.
- `citations[].heading` is a GitHub-style slug of the markdown heading the answer is drawn
  from (see `scripts/slugify.py`) — not a raw chunk id, so it stays valid regardless of how
  any given framework's text splitter draws chunk boundaries. Retrieval metrics (recall@k, MRR)
  are scored by resolving each framework's retrieved chunk up to its enclosing heading section.
- `bucket` definitions:
  - `easy`: answer lives in a single section, direct lookup.
  - `medium`: answer requires synthesizing 2-3 sections.
  - `hard-multihop`: answer requires following a reference chain across sections that don't
    obviously co-occur in a single retrieval pass.
  - `hard-unanswerable`: a plausible-sounding question with no basis in this corpus subset.
    `citations` is `[]`, `unanswerable` is `true`, and `reference_answer` states that the
    corpus doesn't cover it. Tests hallucination discipline independent of retrieval quality.
- Every pair must be hand-verified against the actual vendored text in `corpus/kubernetes-docs/`
  before it ships — no unverified pair belongs in this file.
