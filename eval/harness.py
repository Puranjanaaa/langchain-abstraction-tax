"""
Eval harness: runs eval/qa_set.jsonl against any Implementation and assembles
the metrics from the project spec. Metric bodies live in eval/metrics/ — this
module only owns loading the data, invoking the implementation, and timing
each call (latency has to be measured here, at the call site, not after).
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

from shared.interface import Implementation
from shared.types import Answer

REPO_ROOT = Path(__file__).resolve().parent.parent
QA_SET_PATH = REPO_ROOT / "eval" / "qa_set.jsonl"


@dataclass(frozen=True)
class QAPair:
    id: str
    question: str
    bucket: str
    reference_answer: str
    citations: list[dict]
    unanswerable: bool
    notes: str


@dataclass(frozen=True)
class EvalResult:
    qa: QAPair
    answer: Answer
    latency_s: float


def load_qa_set(path: Path = QA_SET_PATH) -> list[QAPair]:
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(QAPair(**json.loads(line)))
    return pairs


def run(implementation: Implementation, qa_set: list[QAPair] | None = None) -> list[EvalResult]:
    """Runs every question in qa_set through implementation.answer(), timing each call."""
    qa_set = qa_set if qa_set is not None else load_qa_set()
    results = []
    for qa in qa_set:
        start = time.perf_counter()
        answer = implementation.answer(qa.question)
        latency_s = time.perf_counter() - start
        results.append(EvalResult(qa=qa, answer=answer, latency_s=latency_s))
    return results


def score(results: list[EvalResult]) -> dict:
    """Aggregates results into the metrics from the project spec. Each metric
    module is a placeholder as of phase 2 — see eval/metrics/."""
    from eval.metrics import judge, performance, retrieval

    return {
        "retrieval": retrieval.score(results),
        "correctness": judge.score(results),
        "latency": performance.latency(results),
        "cost": performance.cost(results),
    }
