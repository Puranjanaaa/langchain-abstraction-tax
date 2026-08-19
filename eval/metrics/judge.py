"""
LLM-judge scoring for answer correctness and faithfulness: correctness
against `reference_answer`, faithfulness against the cited source text (does
the answer actually follow from what was cited, independent of whether the
citation itself was a good retrieval for the question).
"""

from eval.harness import EvalResult


def score(results: list[EvalResult]) -> dict:
    """Placeholder — LLM-judge scoring is not implemented yet (phase 2 skeleton)."""
    raise NotImplementedError
