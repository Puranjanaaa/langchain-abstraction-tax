import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set — copy .env.example to .env and fill it in")
    return value


def base_url() -> str:
    return _require_env("LLM_BASE_URL")


def api_key() -> str:
    return _require_env("LLM_API_KEY")


def chat_model() -> str:
    return _require_env("LLM_MODEL")


def embedding_model() -> str:
    return _require_env("EMBEDDING_MODEL")


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """A configured OpenAI-compatible client, for implementations that call
    the openai SDK directly (raw API / function-calling) rather than going
    through a framework's own client wrapper."""
    return OpenAI(base_url=base_url(), api_key=api_key())
