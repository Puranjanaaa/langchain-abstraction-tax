"""Plain function-calling implementation of the shared answer() interface:
agentic tool-calling via the openai SDK directly (no orchestration
framework) — the model decides when and how many times to call search_docs
before answering. See agent.py for the loop and why it distrusts message
content on tool-calling turns.
"""

from implementations.function_calling_impl.agent import DECLINE_TEXT, FunctionCallingRAG
from shared.types import Answer, Citation, Usage

name = "function-calling"

_rag: FunctionCallingRAG | None = None


def _get_rag() -> FunctionCallingRAG:
    global _rag
    if _rag is None:
        _rag = FunctionCallingRAG()
    return _rag


def answer(question: str) -> Answer:
    text, raw_responses, chunks = _get_rag().invoke(question)

    citations: list[Citation] = []
    if text.strip() != DECLINE_TEXT:
        seen = set()
        for chunk in chunks:
            key = (chunk.file, chunk.heading)
            if key not in seen:
                seen.add(key)
                citations.append(Citation(file=key[0], heading=key[1]))

    prompt_tokens = 0
    completion_tokens = 0
    have_usage = False
    for response in raw_responses:
        if response.usage:
            have_usage = True
            prompt_tokens += response.usage.prompt_tokens or 0
            completion_tokens += response.usage.completion_tokens or 0

    usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens) if have_usage else None

    return Answer(text=text, citations=citations, usage=usage)
