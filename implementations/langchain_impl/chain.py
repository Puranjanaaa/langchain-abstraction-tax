"""
The LangChain retrieval chain itself: embed the corpus into an in-memory
vector store, retrieve top-k chunks for a question, and generate an answer
grounded in them.

Hand-built from langchain_core primitives (retriever, ChatPromptTemplate,
ChatOpenAI, piped with `|`) rather than langchain_classic's
create_retrieval_chain/create_stuff_documents_chain helpers. In the
installed LangChain version (1.x) those helpers live in langchain_classic, a
legacy-compat package — langchain.chains doesn't exist in the current main
package at all. LCEL composition with `|` is the actual current primitive;
using the classic helpers here would misrepresent what "current LangChain"
looks like, which is the point of this repo.

InMemoryVectorStore (langchain_core) rather than FAISS/Chroma: the corpus is
~6.7k chunks, comfortably fits in memory, and this avoids pulling in a
native-compiled vector store dependency for a one-process benchmark.
"""

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from implementations.langchain_impl.indexing import build_documents
from shared.llm import api_key, base_url, chat_model, embedding_model
from shared.retrieval_config import TOP_K

# The model must say exactly this when the retrieved context doesn't answer
# the question, so answer() can detect the decline programmatically instead
# of pattern-matching free-text refusals.
DECLINE_TEXT = "The provided documentation does not cover this."

SYSTEM_PROMPT = f"""You are a Kubernetes documentation assistant. Answer the \
question using ONLY the context below — do not use outside knowledge, and \
do not guess.

If the context does not contain enough information to answer the question, \
respond with exactly this sentence and nothing else: "{DECLINE_TEXT}"

Context:
{{context}}"""


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata['file']}#{d.metadata['heading']}]\n{d.page_content}" for d in docs
    )


class LangChainRAG:
    def __init__(self) -> None:
        embeddings = OpenAIEmbeddings(
            openai_api_base=base_url(),
            openai_api_key=api_key(),
            model=embedding_model(),
            check_embedding_ctx_length=False,
        )
        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(build_documents())
        self._retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

        llm = ChatOpenAI(
            openai_api_base=base_url(),
            openai_api_key=api_key(),
            model=chat_model(),
            temperature=0,
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "{question}")]
        )
        self._generate = prompt | llm

    def invoke(self, question: str) -> tuple[AIMessage, list[Document]]:
        docs = self._retriever.invoke(question)
        message = self._generate.invoke({"question": question, "context": _format_docs(docs)})
        return message, docs
