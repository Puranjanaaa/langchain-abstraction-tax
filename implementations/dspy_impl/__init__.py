import dspy

from implementations.dspy_impl.module import DECLINE_TEXT, KubernetesRAG
from shared.llm import api_key, base_url, chat_model
from shared.types import Answer, Citation, Usage

name = "dspy"

_rag: KubernetesRAG | None = None


def _get_rag() -> KubernetesRAG:
    global _rag
    if _rag is None:
        lm = dspy.LM(
            f"openai/{chat_model()}",
            api_base=base_url(),
            api_key=api_key(),
            temperature=0,
            cache=False,
        )
        dspy.configure(lm=lm)
        _rag = KubernetesRAG()
    return _rag


def answer(question: str) -> Answer:
    rag = _get_rag()
    prediction, chunks = rag(question)

    # Decline is a structured signal (prediction.answerable), not a string
    # match against DECLINE_TEXT — see module.py's docstring for why. The
    # displayed text is still normalized to DECLINE_TEXT on a decline so
    # transcripts read consistently with the other four implementations.
    declined = not prediction.answerable
    text = DECLINE_TEXT if declined else prediction.answer

    citations: list[Citation] = []
    if not declined:
        seen = set()
        for chunk in chunks:
            key = (chunk.file, chunk.heading)
            if key not in seen:
                seen.add(key)
                citations.append(Citation(file=key[0], heading=key[1]))

    usage = None
    history = dspy.settings.lm.history
    if history:
        raw_usage = history[-1].get("usage") or {}
        if raw_usage:
            usage = Usage(
                prompt_tokens=raw_usage.get("prompt_tokens"),
                completion_tokens=raw_usage.get("completion_tokens"),
            )

    return Answer(text=text, citations=citations, usage=usage)
