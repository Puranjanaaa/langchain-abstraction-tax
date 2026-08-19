"""p50/p95 latency and token usage per query.

"Cost" here means token counts, not a dollar figure: every implementation
currently points at the same local, free LM Studio server (see
shared/llm.py), so there's no real $/token to compute, and inventing one
from an assumed rate would misrepresent a guess as a measurement. Token
counts are still a meaningful, honest axis for the writeup on their own —
e.g. whether a framework's prompt scaffolding costs more tokens than a raw
API call for the same question is real compute overhead regardless of price.
"""

import numpy as np

from eval.harness import EvalResult


def latency(results: list[EvalResult]) -> dict:
    seconds = np.array([r.latency_s for r in results])
    return {
        "p50_s": float(np.percentile(seconds, 50)),
        "p95_s": float(np.percentile(seconds, 95)),
        "n": len(results),
    }


def cost(results: list[EvalResult]) -> dict:
    prompt_tokens = [r.answer.usage.prompt_tokens for r in results if r.answer.usage]
    completion_tokens = [r.answer.usage.completion_tokens for r in results if r.answer.usage]
    prompt_tokens = [t for t in prompt_tokens if t is not None]
    completion_tokens = [t for t in completion_tokens if t is not None]

    def _percentiles(values: list[int]) -> dict:
        if not values:
            return {"p50": None, "p95": None}
        arr = np.array(values)
        return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95))}

    return {
        "prompt_tokens": _percentiles(prompt_tokens),
        "completion_tokens": _percentiles(completion_tokens),
        "n_scored": len(prompt_tokens),
        "n_excluded_no_usage": len(results) - len(prompt_tokens),
        "note": "local model — no $/token; token counts shown as a compute-cost proxy",
    }
