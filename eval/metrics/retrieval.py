"""
Recall@k and MRR, scored at the heading level: a framework's retrieved chunk
is resolved up to its enclosing (file, heading) via corpus/manifest.json
before comparing against a qa pair's citations, so results are comparable
across frameworks regardless of how each one draws chunk boundaries.
"""

from eval.harness import EvalResult


def score(results: list[EvalResult]) -> dict:
    """Placeholder — recall@k and MRR are not implemented yet (phase 2 skeleton)."""
    raise NotImplementedError
