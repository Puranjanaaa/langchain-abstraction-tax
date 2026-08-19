"""DSPy Signature + Module for the generate step. Retrieval (the shared,
locked pipeline — see retrieval.py) is called as a plain function from
forward(), not represented as a dspy.Retrieve component, per the scoping
decision that DSPy's distinctive surface for this benchmark is its typed
Signature/Module generation layer, not a retrieval abstraction duplicating
what's already fixed externally.

No compile/optimizer step (e.g. BootstrapFewShot) is used: compiling
against eval/qa_set.jsonl would blur DSPy's orchestration overhead together
with prompt-optimization gains, which the writeup wants to keep as separate
axes. This is a plain dspy.Predict call, not a dspy.ChainOfThought or
compiled program.

Decline signal: an early version asked the model to emit DECLINE_TEXT
verbatim (mirroring the other four implementations) and gated citations on
an exact string match, the same pattern as langchain_impl. Against the
hard-unanswerable bucket, DSPy's auto-synthesized prompt (built from the
Signature's docstring/field descriptions, not the literal instruction text)
did not reliably reproduce that sentence verbatim — the model paraphrased
instead, which silently broke the "empty citations iff declined" contract
eval/metrics relies on. A typed `answerable` bool field is the idiomatic
DSPy fix: it makes the decline a structured output the adapter parses,
rather than a string pattern hoping the prompt survived DSPy's templating
unchanged. This is itself a real data point on framework behavior, not
just a workaround — see the writeup notes.
"""

import dspy

from implementations.dspy_impl.retrieval import Retriever
from shared.chunking import Chunk
from shared.retrieval_config import TOP_K

DECLINE_TEXT = "The provided documentation does not cover this."


class AnswerFromContext(dspy.Signature):
    """Answer the Kubernetes documentation question using ONLY the provided context — do not use outside \
knowledge, and do not guess."""

    context: str = dspy.InputField(desc="Kubernetes documentation excerpts relevant to the question")
    question: str = dspy.InputField()
    answerable: bool = dspy.OutputField(
        desc="True if the context contains enough information to answer the question; "
        "False if it does not, and the question should be declined rather than guessed at"
    )
    answer: str = dspy.OutputField(
        desc="The answer, grounded only in the context. Leave empty if answerable is False."
    )


def _format_chunks(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.file}#{c.heading}]\n{c.text}" for c in chunks)


class KubernetesRAG(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self._retriever = Retriever()
        # ChainOfThought rather than bare Predict: on this local 8B model, a
        # bare Predict was observed setting answerable=True on a question
        # whose own `answer` field said the context didn't cover it — a
        # self-contradiction. Giving the model a reasoning field before
        # committing to the verdict resolved that specific case; it isn't a
        # guarantee against all such inconsistency from a small local model.
        self.generate = dspy.ChainOfThought(AnswerFromContext)

    def forward(self, question: str) -> tuple[dspy.Prediction, list[Chunk]]:
        chunks = self._retriever.retrieve(question, TOP_K)
        prediction = self.generate(context=_format_chunks(chunks), question=question)
        return prediction, chunks
