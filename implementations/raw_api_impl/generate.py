from implementations.raw_api_impl.retrieval import Chunk, Retriever
from shared.llm import chat_model, get_client
from shared.retrieval_config import TOP_K

DECLINE_TEXT = "The provided documentation does not cover this."

SYSTEM_PROMPT = f"""You are a Kubernetes documentation assistant. Answer the \
question using ONLY the context below — do not use outside knowledge, and \
do not guess.

If the context does not contain enough information to answer the question, \
respond with exactly this sentence and nothing else: "{DECLINE_TEXT}"

Context:
{{context}}"""


def _format_chunks(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.file}#{c.heading}]\n{c.text}" for c in chunks)


class RawAPIRAG:
    def __init__(self) -> None:
        self._retriever = Retriever()

    def invoke(self, question: str) -> tuple[str, object, list[Chunk]]:
        chunks = self._retriever.retrieve(question, TOP_K)
        response = get_client().chat.completions.create(
            model=chat_model(),
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(context=_format_chunks(chunks))},
                {"role": "user", "content": question},
            ],
        )
        text = response.choices[0].message.content or ""
        return text, response, chunks
