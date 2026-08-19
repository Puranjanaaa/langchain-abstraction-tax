"""LlamaIndex implementation of the shared answer() interface.

Retrieval setup (chunking, top-k) is locked in shared/retrieval_config.py
and held identical across all five implementations via shared/chunking.py
— see implementations/llamaindex_impl/indexing.py for how those chunks
become a VectorStoreIndex and query.py for the retrieve-then-generate loop.
"""

from implementations.llamaindex_impl.query import DECLINE_TEXT, LlamaIndexRAG
from shared.types import Answer, Citation, Usage

name = "llamaindex"

_rag: LlamaIndexRAG | None = None


def _get_rag() -> LlamaIndexRAG:
    global _rag
    if _rag is None:
        _rag = LlamaIndexRAG()
    return _rag


def answer(question: str) -> Answer:
    text, raw_response, nodes = _get_rag().invoke(question)

    citations: list[Citation] = []
    if text.strip() != DECLINE_TEXT:
        seen = set()
        for node in nodes:
            key = (node.node.metadata["file"], node.node.metadata["heading"])
            if key not in seen:
                seen.add(key)
                citations.append(Citation(file=key[0], heading=key[1]))

    usage = None
    if raw_response is not None and getattr(raw_response, "usage", None):
        usage = Usage(
            prompt_tokens=raw_response.usage.prompt_tokens,
            completion_tokens=raw_response.usage.completion_tokens,
        )

    return Answer(text=text, citations=citations, usage=usage)
