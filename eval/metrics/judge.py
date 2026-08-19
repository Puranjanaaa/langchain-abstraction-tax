import re

from eval.harness import EvalResult
from shared.llm import chat_model, get_client
from shared.source_text import resolve

_CORRECTNESS_SYSTEM = """You are grading a Kubernetes documentation Q&A system. \
Compare the CANDIDATE ANSWER against the REFERENCE ANSWER for the same question \
and classify it as exactly one of:

- "correct": conveys the same information as the reference. If the reference \
says the corpus doesn't cover this question, a candidate that declines to \
answer (rather than guessing) is "correct".
- "partially_correct": gets the core point right but is incomplete, or \
includes a minor inaccuracy alongside otherwise correct content.
- "incorrect": contradicts the reference, answers a different question, \
fabricates an answer to a question the reference says is unanswerable, or \
declines to answer a question the reference says IS answerable.

Respond with ONLY a JSON object: {"correctness": "correct" | "partially_correct" \
| "incorrect", "reasoning": "<one sentence>"}"""

_FAITHFULNESS_SYSTEM = """You are checking whether an answer is faithful to its \
cited source text — i.e. whether every factual claim in the ANSWER is supported \
by the CITED SOURCE TEXT, regardless of whether that source text was a good \
choice to cite for the question. Ignore completeness; judge only whether what's \
stated is supported by the source.

Paraphrasing, summarizing, rewording, and combining information from different \
parts of the source are all faithful — this is expected, not a violation. Judge \
"unfaithful" only when the answer states something the source doesn't support: a \
fact the source doesn't contain, a claim that contradicts the source, or a detail \
invented beyond what the source says. A rephrased sentence that means the same \
thing as the source is faithful even if no wording matches verbatim.

Respond with ONLY a JSON object: {"faithfulness": "faithful" | "unfaithful", \
"reasoning": "<one sentence>"}"""


def _call_judge(system: str, user: str) -> str:
    response = get_client().chat.completions.create(
        model=chat_model(),
        temperature=0,
        max_tokens=200,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return response.choices[0].message.content or ""


def _extract_field(content: str, field: str, allowed: tuple[str, ...]) -> str:
    """Regex-extracts a field's value directly rather than requiring a fully
    valid JSON parse — local models reliably emit the `"field": "value"` pair
    itself (it's instructed first in the schema) but sometimes trail off or
    add stray text afterward instead of closing the object cleanly."""
    match = re.search(rf'"{field}"\s*:\s*"({"|".join(allowed)})"', content)
    if not match:
        raise ValueError(f"could not find {field!r} in judge output: {content!r}")
    return match.group(1)


def _judge_correctness(result: EvalResult) -> str:
    user = (
        f"QUESTION:\n{result.qa.question}\n\n"
        f"REFERENCE ANSWER:\n{result.qa.reference_answer}\n\n"
        f"CANDIDATE ANSWER:\n{result.answer.text}"
    )
    content = _call_judge(_CORRECTNESS_SYSTEM, user)
    return _extract_field(content, "correctness", ("correct", "partially_correct", "incorrect"))


def _resolve_cited_text(citations: list) -> tuple[str, list[str]]:
    """Returns (concatenated source text, list of citations that failed to
    resolve). A citation that doesn't resolve points at a (file, heading)
    the manifest doesn't have — a bogus citation, which is itself a
    faithfulness problem, not a lookup error to hide."""
    chunks = []
    broken = []
    for citation in citations:
        try:
            text = resolve(citation.file, citation.heading)
            chunks.append(f"[{citation.file}#{citation.heading}]\n{text}")
        except KeyError:
            broken.append(f"{citation.file}#{citation.heading}")
    return "\n\n".join(chunks), broken


def _judge_faithfulness(result: EvalResult) -> str:
    cited_text, broken = _resolve_cited_text(result.answer.citations)
    if not cited_text:
        return "unfaithful"  # every citation was bogus — nothing grounds the answer

    user = f"CITED SOURCE TEXT:\n{cited_text}\n\nANSWER:\n{result.answer.text}"
    if broken:
        user += f"\n\n(Note: {len(broken)} additional citation(s) didn't resolve to a real corpus section: {broken})"

    content = _call_judge(_FAITHFULNESS_SYSTEM, user)
    return _extract_field(content, "faithfulness", ("faithful", "unfaithful"))


def score(results: list[EvalResult]) -> dict:
    correctness_counts = {"correct": 0, "partially_correct": 0, "incorrect": 0}
    faithfulness_counts = {"faithful": 0, "unfaithful": 0}
    faithfulness_scored = 0
    per_result = []

    for r in results:
        correctness = _judge_correctness(r)
        correctness_counts[correctness] += 1

        if r.answer.citations:
            faithfulness = _judge_faithfulness(r)
            faithfulness_counts[faithfulness] += 1
            faithfulness_scored += 1
        else:
            faithfulness = "not_applicable"

        per_result.append({"id": r.qa.id, "correctness": correctness, "faithfulness": faithfulness})

    n = len(results)
    return {
        "correctness": {
            "correct_rate": correctness_counts["correct"] / n,
            "partially_correct_rate": correctness_counts["partially_correct"] / n,
            "incorrect_rate": correctness_counts["incorrect"] / n,
            "n": n,
        },
        "faithfulness": {
            "faithful_rate": (
                faithfulness_counts["faithful"] / faithfulness_scored if faithfulness_scored else None
            ),
            "unfaithful_rate": (
                faithfulness_counts["unfaithful"] / faithfulness_scored if faithfulness_scored else None
            ),
            "n_scored": faithfulness_scored,
            "n_excluded_no_citations": n - faithfulness_scored,
        },
        "per_result": per_result,
    }
