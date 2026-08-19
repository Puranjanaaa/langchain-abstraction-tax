import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import load_qa_set
from implementations import (
    dspy_impl,
    function_calling_impl,
    langchain_impl,
    llamaindex_impl,
    raw_api_impl,
)
from shared.llm import base_url, chat_model, embedding_model, get_client
from shared.types import Answer, Citation


def section(title: str) -> None:
    print(f"\n--- {title} ---")


def main() -> None:
    section("env config (shared/llm.py)")
    print(f"LLM_BASE_URL:     {base_url()}")
    print(f"LLM_MODEL:        {chat_model()}")
    print(f"EMBEDDING_MODEL:  {embedding_model()}")

    section("live chat completion")
    client = get_client()
    resp = client.chat.completions.create(
        model=chat_model(),
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        max_tokens=10,
    )
    print(f"response: {resp.choices[0].message.content!r}")

    section("live embedding call")
    emb = client.embeddings.create(model=embedding_model(), input="hello world")
    print(f"embedding dims: {len(emb.data[0].embedding)}")

    section("eval set (eval/harness.py)")
    qa_set = load_qa_set()
    print(f"loaded {len(qa_set)} qa pairs")
    sample = qa_set[0]
    print(f"sample: [{sample.id}] ({sample.bucket}) {sample.question}")

    section("shared types (shared/types.py)")
    answer = Answer(
        text="maxUnavailable defaults to 25%.",
        citations=[Citation(file="concepts/workloads/controllers/deployment.md", heading="max-unavailable")],
    )
    print(answer)

    section("implementations (implementations/)")
    for impl in (langchain_impl, llamaindex_impl, dspy_impl, function_calling_impl, raw_api_impl):
        try:
            answer = impl.answer("What does a Kubernetes Deployment do?")
        except NotImplementedError:
            print(f"{impl.name}: not implemented yet")
        else:
            print(f"{impl.name}: {answer.text[:80]!r} ({len(answer.citations)} citations)")

    print("\nsmoke test passed: every piece is wired up correctly.")


if __name__ == "__main__":
    main()
