"""The shared interface every RAG implementation (LangChain, LlamaIndex, DSPy,
function-calling, raw API) satisfies, so the eval harness runs identically
against all five — the orchestration layer is the only thing that varies.
"""

from typing import Protocol

from shared.types import Answer


class Implementation(Protocol):
    name: str

    def answer(self, question: str) -> Answer: ...
