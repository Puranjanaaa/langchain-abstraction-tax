from typing import Protocol

from shared.types import Answer


class Implementation(Protocol):
    name: str

    def answer(self, question: str) -> Answer: ...
