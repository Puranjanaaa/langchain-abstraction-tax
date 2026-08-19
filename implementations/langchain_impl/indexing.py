import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared.retrieval_config import CHUNK_OVERLAP, SECTION_CHAR_LIMIT

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = REPO_ROOT / "corpus" / "kubernetes-docs"
MANIFEST_PATH = REPO_ROOT / "corpus" / "manifest.json"

_HEADING_LINE_RE = re.compile(r"^#{1,6}[ \t]+.+?[ \t]*$", re.MULTILINE)

_sub_splitter = RecursiveCharacterTextSplitter(
    chunk_size=SECTION_CHAR_LIMIT,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _load_manifest_files() -> list[dict]:
    return json.loads(MANIFEST_PATH.read_text())["files"]


def _split_into_sections(text: str, heading_slugs: list[str]) -> list[tuple[str, str]]:
    """Returns [(heading_slug, section_text), ...] in document order.

    A section runs from its heading line up to the next heading line (of any
    level) or end of file, so it includes that heading's own body content
    but not a subsection's trailing sibling.
    """
    starts = [m.start() for m in _HEADING_LINE_RE.finditer(text)]
    assert len(starts) == len(heading_slugs), "heading-line count must match manifest count"

    sections = []
    for i, slug in enumerate(heading_slugs):
        start = starts[i]
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        sections.append((slug, text[start:end].strip()))
    return sections


def build_documents() -> list[Document]:
    """Builds the flat list of chunks to embed and index, one pass over the
    whole corpus. Cheap enough (396 files) to call once at process start."""
    documents: list[Document] = []
    for file_entry in _load_manifest_files():
        rel_path = file_entry["path"]
        slugs = [h["slug"] for h in file_entry["headings"]]
        text = (CORPUS_DIR / rel_path).read_text()
        for slug, section_text in _split_into_sections(text, slugs):
            metadata = {"file": rel_path, "heading": slug}
            if len(section_text) <= SECTION_CHAR_LIMIT:
                documents.append(Document(page_content=section_text, metadata=metadata))
            else:
                for chunk in _sub_splitter.split_text(section_text):
                    documents.append(Document(page_content=chunk, metadata=metadata))
    return documents
