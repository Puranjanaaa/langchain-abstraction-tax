"""Raw API implementation of the shared answer() interface (direct HTTP calls,
no SDK, no framework). Not yet implemented."""

from shared.types import Answer

name = "raw-api"


def answer(question: str) -> Answer:
    raise NotImplementedError
