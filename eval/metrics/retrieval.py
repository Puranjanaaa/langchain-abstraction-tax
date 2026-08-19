"""
Recall@k and MRR, scored at the heading level: a framework's retrieved chunk
is resolved up to its enclosing (file, heading) via corpus/manifest.json
before comparing against a qa pair's citations, so results are comparable
across frameworks regardless of how each one draws chunk boundaries.

`k` isn't a parameter here — Answer.citations already *is* the
implementation's deduped, rank-ordered top-k retrieval (see
shared/retrieval_config.TOP_K, held identical across implementations), so
these metrics just consume it as given.

hard-unanswerable pairs (qa.citations == []) are excluded from both
averages: an empty expected-citation set makes "did we retrieve the right
thing" undefined, not zero. Whether an implementation correctly declined
those questions is judged separately (see eval/metrics/judge.py) — folding
it in here would conflate retrieval quality with generation behavior.
"""

from eval.harness import EvalResult


def _recall_and_rr(qa_citations: list[dict], retrieved: list[tuple[str, str]]) -> tuple[float, float]:
    expected = {(c["file"], c["heading"]) for c in qa_citations}
    hits = expected & set(retrieved)
    recall = len(hits) / len(expected)

    reciprocal_rank = 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in expected:
            reciprocal_rank = 1.0 / rank
            break

    return recall, reciprocal_rank


def score(results: list[EvalResult]) -> dict:
    scorable = [r for r in results if r.qa.citations]
    excluded = len(results) - len(scorable)

    recalls = []
    reciprocal_ranks = []
    for r in scorable:
        retrieved = [(c.file, c.heading) for c in r.answer.citations]
        recall, rr = _recall_and_rr(r.qa.citations, retrieved)
        recalls.append(recall)
        reciprocal_ranks.append(rr)

    return {
        "recall_at_k": sum(recalls) / len(recalls) if recalls else None,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None,
        "n_scored": len(scorable),
        "n_excluded_unanswerable": excluded,
    }
