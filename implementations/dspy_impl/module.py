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
