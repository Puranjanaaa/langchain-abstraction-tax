from implementations.raw_api_impl.generate import DECLINE_TEXT, RawAPIRAG
from shared.types import Answer, Citation, Usage

name = "raw-api"

_rag: RawAPIRAG | None = None


def _get_rag() -> RawAPIRAG:
    global _rag
    if _rag is None:
        _rag = RawAPIRAG()
    return _rag


def answer(question: str) -> Answer:
    text, response, chunks = _get_rag().invoke(question)

    citations: list[Citation] = []
    if text.strip() != DECLINE_TEXT:
        seen = set()
        for chunk in chunks:
            key = (chunk.file, chunk.heading)
            if key not in seen:
                seen.add(key)
                citations.append(Citation(file=key[0], heading=key[1]))

    usage = None
    if response.usage:
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens, completion_tokens=response.usage.completion_tokens
        )

    return Answer(text=text, citations=citations, usage=usage)
