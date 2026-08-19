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
