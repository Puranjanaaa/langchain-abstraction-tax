"""Agentic function-calling loop: the model is given a search_docs tool and
decides for itself when, how, and how many times to call it before
answering, rather than a fixed retrieve-then-generate sequence. This is
what's distinctive about "function-calling as orchestration" for the
writeup — retrieval internals (embed query, top-k, chunking) still call the
exact same locked pipeline as every other implementation (see retrieval.py),
only the decision of *when* to invoke it is left to the model.

The local model (see shared/llm.py) has an observed quirk worth guarding
against explicitly: on a turn where it emits a real tool_calls entry, its
`message.content` sometimes also contains a fabricated, hallucinated
"[TOOL_RESULT]...[END_TOOL_RESULT]" block — invented before the real tool
result was ever returned. That content must never be treated as part of the
answer. Only a turn with no tool_calls is trusted as the final answer.
"""

import json

from openai.types.chat import ChatCompletionMessageParam

from implementations.function_calling_impl.retrieval import Chunk, Retriever
from shared.llm import chat_model, get_client
from shared.retrieval_config import TOP_K

DECLINE_TEXT = "The provided documentation does not cover this."
MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = f"""You are a Kubernetes documentation assistant. You do not \
know anything about Kubernetes from your own training — you must call the \
search_docs tool to look up relevant documentation before answering. You may \
call it more than once with different queries if the question has multiple \
parts. Answer using ONLY what search_docs returns — do not use outside \
knowledge, and do not guess.

Once you have enough information, respond with your final answer as plain \
text (no further tool call). If what search_docs returned does not contain \
enough information to answer the question, respond with exactly this \
sentence and nothing else: "{DECLINE_TEXT}\""""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the Kubernetes documentation corpus for sections relevant to a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search the documentation for"}
                },
                "required": ["query"],
            },
        },
    }
]


def _format_chunks(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.file}#{c.heading}]\n{c.text}" for c in chunks)


class FunctionCallingRAG:
    def __init__(self) -> None:
        self._retriever = Retriever()

    def invoke(self, question: str) -> tuple[str, list[dict], list[Chunk]]:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        all_chunks: list[Chunk] = []
        raw_responses: list[dict] = []

        for _ in range(MAX_TOOL_ROUNDS):
            response = get_client().chat.completions.create(
                model=chat_model(), temperature=0, messages=messages, tools=TOOLS, tool_choice="auto"
            )
            raw_responses.append(response)
            message = response.choices[0].message

            if not message.tool_calls:
                return message.content or "", raw_responses, all_chunks

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in message.tool_calls
                    ],
                }
            )
            for tool_call in message.tool_calls:
                try:
                    query = json.loads(tool_call.function.arguments)["query"]
                except (json.JSONDecodeError, KeyError):
                    query = question
                chunks = self._retriever.retrieve(query, TOP_K)
                all_chunks.extend(chunks)
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": _format_chunks(chunks)}
                )

        # Exhausted MAX_TOOL_ROUNDS without a final answer — force one more
        # call with tools disabled so the model must produce plain text
        # rather than looping forever.
        response = get_client().chat.completions.create(model=chat_model(), temperature=0, messages=messages)
        raw_responses.append(response)
        return response.choices[0].message.content or "", raw_responses, all_chunks
