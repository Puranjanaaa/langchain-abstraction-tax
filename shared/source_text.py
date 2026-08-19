"""Resolves a citation's (file, heading) into the section text it actually
points at, via corpus/manifest.json. Used by eval/metrics/judge.py to check
whether an answer is faithful to what its citations actually say.

Same heading-splitting algorithm as
implementations/langchain_impl/indexing.py's _split_into_sections (a section
runs from its heading line to the next heading line or end of file). Factored
out here rather than left private to that implementation because every
implementation's citations resolve to (file, heading) pairs per
eval/SCHEMA.md, so all five need identical text for the same citation —
this is eval-side infrastructure, not LangChain-specific.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "corpus" / "kubernetes-docs"
MANIFEST_PATH = REPO_ROOT / "corpus" / "manifest.json"

_HEADING_LINE_RE = re.compile(r"^#{1,6}[ \t]+.+?[ \t]*$", re.MULTILINE)


@lru_cache(maxsize=1)
def _sections_by_file() -> dict[str, dict[str, str]]:
    """{file: {heading_slug: section_text}}, built once from the manifest."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    result: dict[str, dict[str, str]] = {}
    for entry in manifest["files"]:
        rel_path = entry["path"]
        slugs = [h["slug"] for h in entry["headings"]]
        text = (CORPUS_DIR / rel_path).read_text()
        starts = [m.start() for m in _HEADING_LINE_RE.finditer(text)]
        assert len(starts) == len(slugs), f"heading-line count must match manifest count for {rel_path}"

        sections = {}
        for i, slug in enumerate(slugs):
            start = starts[i]
            end = starts[i + 1] if i + 1 < len(starts) else len(text)
            sections[slug] = text[start:end].strip()
        result[rel_path] = sections
    return result


def resolve(file: str, heading: str) -> str:
    """Returns the section text a citation's (file, heading) points at.

    Raises KeyError if the file or heading isn't in the manifest — that
    means the citation doesn't correspond to a real corpus section.
    """
    sections = _sections_by_file()
    if file not in sections:
        raise KeyError(f"unknown file in citation: {file}")
    if heading not in sections[file]:
        raise KeyError(f"unknown heading in citation: {file}#{heading}")
    return sections[file][heading]
