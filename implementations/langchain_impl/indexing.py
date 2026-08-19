"""
Turns corpus/kubernetes-docs/*.md into LangChain Documents for the vector
store. Chunking algorithm (locked across all five implementations — see
shared/retrieval_config.py):

  1. Split each file on its markdown headings first, so every chunk maps to
     exactly one heading by construction. eval/SCHEMA.md resolves citations
     at heading granularity regardless of how a framework draws chunk
     boundaries; splitting on headings up front means that resolution never
     has to guess which heading a chunk's tail belongs to.
  2. Any section longer than SECTION_CHAR_LIMIT gets recursively sub-split,
     with sub-chunks inheriting the parent section's (file, heading)
     metadata.

Heading slugs come from corpus/manifest.json rather than being re-derived
from the vendored markdown. scripts/vendor_corpus.py strips explicit `{#id}`
anchor overrides from the on-disk body for clean display (~16% of upstream
headings carry one) — that id only survives in the manifest, computed once
at vendor time from the pre-strip source. Re-slugifying the on-disk text
would silently get those headings wrong; manifest.json is the single source
of truth for citation-matching slugs (see eval/SCHEMA.md). Each file's
manifest heading order is matched positionally against its heading lines —
verified 1:1 across all 396 files before relying on this.
"""

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
