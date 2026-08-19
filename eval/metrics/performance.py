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
