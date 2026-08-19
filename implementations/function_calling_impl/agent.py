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
