"""Retrieve top-k nodes for a question via a LlamaIndex retriever, then
generate an answer grounded in them through SharedClientLLM. Hand-built
retrieve-then-generate rather than LlamaIndex's RetrieverQueryEngine /
response-synthesizer machinery, so the prompt actually sent to the model is
visible and directly comparable to the other four implementations' prompts.
"""

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore

from implementations.llamaindex_impl.indexing import build_index, build_retriever
from implementations.llamaindex_impl.llm import SharedClientLLM

DECLINE_TEXT = "The provided documentation does not cover this."

SYSTEM_PROMPT = f"""You are a Kubernetes documentation assistant. Answer the \
question using ONLY the context below — do not use outside knowledge, and \
do not guess.

If the context does not contain enough information to answer the question, \
respond with exactly this sentence and nothing else: "{DECLINE_TEXT}"

Context:
{{context}}"""


def _format_nodes(nodes: list[NodeWithScore]) -> str:
    return "\n\n".join(
        f"[{n.node.metadata['file']}#{n.node.metadata['heading']}]\n{n.node.get_content()}"
        for n in nodes
    )


class LlamaIndexRAG:
    def __init__(self) -> None:
        self._retriever = build_retriever(build_index())
        self._llm = SharedClientLLM()

    def invoke(self, question: str) -> tuple[str, object, list[NodeWithScore]]:
        nodes = self._retriever.retrieve(question)
        prompt = SYSTEM_PROMPT.format(context=_format_nodes(nodes))
        response = self._llm.chat(
            [
                ChatMessage(role=MessageRole.SYSTEM, content=prompt),
                ChatMessage(role=MessageRole.USER, content=question),
            ]
        )
        return response.message.content or "", response.raw, nodes
