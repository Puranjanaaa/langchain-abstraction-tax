"""p50/p95 latency and token cost per query."""

from eval.harness import EvalResult


def latency(results: list[EvalResult]) -> dict:
    """Placeholder — p50/p95 latency aggregation is not implemented yet (phase 2 skeleton)."""
    raise NotImplementedError


def cost(results: list[EvalResult]) -> dict:
    """Placeholder — token cost aggregation is not implemented yet (phase 2 skeleton).

    Depends on each Answer's usage field, which is populated only when the
    provider reports token counts (see shared/types.py).
    """
    raise NotImplementedError
