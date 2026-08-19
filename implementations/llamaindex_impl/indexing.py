from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode

from implementations.llamaindex_impl.llm import SharedClientEmbedding
from shared.chunking import build_chunks
from shared.retrieval_config import TOP_K


def build_index() -> VectorStoreIndex:
    nodes = [
        TextNode(text=chunk.text, metadata={"file": chunk.file, "heading": chunk.heading})
        for chunk in build_chunks()
    ]
    return VectorStoreIndex(nodes=nodes, embed_model=SharedClientEmbedding())


def build_retriever(index: VectorStoreIndex):
    return index.as_retriever(similarity_top_k=TOP_K)
