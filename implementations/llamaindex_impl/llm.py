"""Wraps shared.llm's OpenAI-compatible client as LlamaIndex's LLM and
BaseEmbedding interfaces, so llamaindex_impl gets real LlamaIndex indexing/
retrieval machinery (VectorStoreIndex, retrievers) while every network call
still goes through the one shared, env-configured client — never
LlamaIndex's own openai-specific integration classes, which would configure
the endpoint a second, independent way (and, as of this writing, pin an
`openai<3` constraint that conflicts with this repo's `openai>=3.3.0`).

No streaming or true async is implemented — the eval harness only ever
makes blocking answer() calls, so the stream_*/a* methods LlamaIndex's LLM
ABC requires are thin synchronous wrappers, not real implementations.
"""

from typing import Any, Sequence

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.llm import LLM

from shared.llm import chat_model, embedding_model, get_client

_ROLE_TO_OPENAI = {
    MessageRole.SYSTEM: "system",
    MessageRole.USER: "user",
    MessageRole.ASSISTANT: "assistant",
}


class SharedClientLLM(LLM):
    context_window: int = 8192
    num_output: int = 1024

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=chat_model(),
            is_chat_model=True,
        )

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        openai_messages = [
            {"role": _ROLE_TO_OPENAI.get(m.role, "user"), "content": m.content or ""}
            for m in messages
        ]
        response = get_client().chat.completions.create(
            model=chat_model(), temperature=0, messages=openai_messages
        )
        content = response.choices[0].message.content or ""
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=content), raw=response
        )

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        chat_response = self.chat([ChatMessage(role=MessageRole.USER, content=prompt)])
        return CompletionResponse(text=chat_response.message.content or "", raw=chat_response.raw)

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        response = self.chat(messages, **kwargs)

        def gen() -> ChatResponseGen:
            yield response

        return gen()

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        response = self.complete(prompt, formatted=formatted, **kwargs)

        def gen() -> CompletionResponseGen:
            yield response

        return gen()

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self.chat(messages, **kwargs)

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return self.complete(prompt, formatted=formatted, **kwargs)

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        response = self.chat(messages, **kwargs)

        async def gen() -> ChatResponseAsyncGen:
            yield response

        return gen()

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        response = self.complete(prompt, formatted=formatted, **kwargs)

        async def gen() -> CompletionResponseAsyncGen:
            yield response

        return gen()


class SharedClientEmbedding(BaseEmbedding):
    def __init__(self, **kwargs: Any) -> None:
        # LlamaIndex's default embed_batch_size (10) means ~670 sequential
        # HTTP round-trips to embed the 6.7k-chunk corpus; a larger batch
        # cuts that to a fraction without changing which vectors are
        # produced, since embedding is batch-size-independent.
        kwargs.setdefault("embed_batch_size", 100)
        super().__init__(model_name=embedding_model(), **kwargs)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = get_client().embeddings.create(model=embedding_model(), input=texts)
        return [item.embedding for item in response.data]
