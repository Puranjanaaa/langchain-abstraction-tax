from implementations.langchain_impl.chain import DECLINE_TEXT, LangChainRAG
from shared.types import Answer, Citation, Usage

name = "langchain"

_rag: LangChainRAG | None = None


def _get_rag() -> LangChainRAG:
    global _rag
    if _rag is None:
        _rag = LangChainRAG()
    return _rag


def answer(question: str) -> Answer:
    message, docs = _get_rag().invoke(question)
    text = message.content

    citations: list[Citation] = []
    if text.strip() != DECLINE_TEXT:
        seen = set()
        for doc in docs:
            key = (doc.metadata["file"], doc.metadata["heading"])
            if key not in seen:
                seen.add(key)
                citations.append(Citation(file=key[0], heading=key[1]))

    usage = None
    if message.usage_metadata:
        usage = Usage(
            prompt_tokens=message.usage_metadata.get("input_tokens"),
            completion_tokens=message.usage_metadata.get("output_tokens"),
        )

    return Answer(text=text, citations=citations, usage=usage)
