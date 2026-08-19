"""Framework-agnostic header-aware chunking, extracted in Phase 5 so all five
implementations produce byte-identical chunk boundaries regardless of which
framework's Document/Node type wraps them.

Two-stage algorithm (parameters locked in shared/retrieval_config.py):
  1. Split each file on its markdown headings, so every chunk maps to
     exactly one heading by construction. Heading slugs come from
     corpus/manifest.json rather than being re-derived from the vendored
     markdown — see the note in implementations/langchain_impl/indexing.py
     for why (stripped `{#id}` anchor overrides).
  2. Any section longer than SECTION_CHAR_LIMIT gets recursively sub-split,
     with sub-chunks inheriting the parent section's (file, heading).

implementations/langchain_impl/indexing.py predates this module (Phase 3)
and is locked, so it keeps its own private copy of stage 1 backed by
langchain_text_splitters directly for stage 2. The sub-split logic below is
a from-scratch port of RecursiveCharacterTextSplitter's algorithm
(separators=["\\n\\n", "\\n", ". ", " ", ""], keep_separator=True) rather
than an import of that package, specifically so implementations/raw_api_impl
(meant to carry zero framework code) and the other three don't pick up a
langchain dependency transitively through shared/. Verified to produce
byte-identical output to the real langchain_text_splitters implementation
across every oversized section in the corpus.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from shared.retrieval_config import CHUNK_OVERLAP, SECTION_CHAR_LIMIT

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "corpus" / "kubernetes-docs"
MANIFEST_PATH = REPO_ROOT / "corpus" / "manifest.json"

_HEADING_LINE_RE = re.compile(r"^#{1,6}[ \t]+.+?[ \t]*$", re.MULTILINE)
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True)
class Chunk:
    file: str
    heading: str
    text: str


def _load_manifest_files() -> list[dict]:
    return json.loads(MANIFEST_PATH.read_text())["files"]


def split_into_sections(text: str, heading_slugs: list[str]) -> list[tuple[str, str]]:
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


def _split_with_separator(text: str, separator: str) -> list[str]:
    """Splits on separator, gluing it to the start of each following piece —
    mirrors RecursiveCharacterTextSplitter's default keep_separator=True
    ('start') behavior."""
    if not separator:
        return list(text)
    parts = re.split(f"({re.escape(separator)})", text)
    merged = [parts[i] + parts[i + 1] for i in range(1, len(parts), 2)]
    if len(parts) % 2 == 0:
        merged += parts[-1:]
    pieces = [parts[0], *merged]
    return [p for p in pieces if p]


def _join(pieces: list[str]) -> str | None:
    text = "".join(pieces).strip()
    return text or None


def _merge_pieces(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    # keep_separator=True means the separator travels with each piece
    # already, so the merge join separator is always "" — no separator_len
    # bookkeeping needed (unlike the general TextSplitter._merge_splits).
    chunks = []
    current: list[str] = []
    total = 0
    for piece in pieces:
        length = len(piece)
        if total + length > chunk_size:
            if current:
                joined = _join(current)
                if joined is not None:
                    chunks.append(joined)
                while total > chunk_overlap or (total + length > chunk_size and total > 0):
                    total -= len(current[0])
                    current = current[1:]
        current.append(piece)
        total += length
    joined = _join(current)
    if joined is not None:
        chunks.append(joined)
    return chunks


def _recursive_split(text: str, separators: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    separator = separators[-1]
    remaining_separators: list[str] = []
    for i, candidate in enumerate(separators):
        if not candidate:
            separator = candidate
            break
        if re.search(re.escape(candidate), text):
            separator = candidate
            remaining_separators = separators[i + 1 :]
            break

    pieces = _split_with_separator(text, separator)

    final_chunks = []
    good_pieces: list[str] = []
    for piece in pieces:
        if len(piece) < chunk_size:
            good_pieces.append(piece)
        else:
            if good_pieces:
                final_chunks.extend(_merge_pieces(good_pieces, chunk_size, chunk_overlap))
                good_pieces = []
            if not remaining_separators:
                final_chunks.append(piece)
            else:
                final_chunks.extend(_recursive_split(piece, remaining_separators, chunk_size, chunk_overlap))
    if good_pieces:
        final_chunks.extend(_merge_pieces(good_pieces, chunk_size, chunk_overlap))
    return final_chunks


def sub_split(section_text: str) -> list[str]:
    """Sub-splits a section longer than SECTION_CHAR_LIMIT into overlapping
    chunks, using the locked chunk_size/overlap from retrieval_config."""
    return _recursive_split(section_text, _SEPARATORS, SECTION_CHAR_LIMIT, CHUNK_OVERLAP)


def build_chunks() -> list[Chunk]:
    """One pass over the whole corpus, producing the flat list of (file,
    heading, text) chunks every implementation embeds and indexes. Cheap
    enough (396 files) to call once at process start."""
    chunks: list[Chunk] = []
    for file_entry in _load_manifest_files():
        rel_path = file_entry["path"]
        slugs = [h["slug"] for h in file_entry["headings"]]
        text = (CORPUS_DIR / rel_path).read_text()
        for slug, section_text in split_into_sections(text, slugs):
            if len(section_text) <= SECTION_CHAR_LIMIT:
                chunks.append(Chunk(file=rel_path, heading=slug, text=section_text))
            else:
                for sub_text in sub_split(section_text):
                    chunks.append(Chunk(file=rel_path, heading=slug, text=sub_text))
    return chunks
