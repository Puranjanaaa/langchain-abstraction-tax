import numpy as np

from shared.chunking import Chunk, build_chunks
from shared.llm import embedding_model, get_client
from shared.retrieval_config import TOP_K

_EMBED_BATCH = 100


def _embed(texts: list[str]) -> np.ndarray:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH):
        batch = texts[i : i + _EMBED_BATCH]
        response = get_client().embeddings.create(model=embedding_model(), input=batch)
        vectors.extend(item.embedding for item in response.data)
    matrix = np.array(vectors, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


class Retriever:
    def __init__(self) -> None:
        self._chunks = build_chunks()
        self._embeddings = _embed([c.text for c in self._chunks])

    def retrieve(self, query: str, k: int = TOP_K) -> list[Chunk]:
        query_vec = _embed([query])[0]
        scores = self._embeddings @ query_vec
        top_indices = np.argsort(-scores)[:k]
        return [self._chunks[i] for i in top_indices]
