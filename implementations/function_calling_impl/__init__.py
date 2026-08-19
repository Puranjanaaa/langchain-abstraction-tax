"""Plain function-calling implementation of the shared answer() interface (openai SDK,
no orchestration framework). Not yet implemented."""

from shared.types import Answer

name = "function-calling"


def answer(question: str) -> Answer:
    raise NotImplementedError
