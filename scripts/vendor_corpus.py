import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from headings import extract_headings, strip_explicit_ids_from_body

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_REPO = "https://github.com/kubernetes/website.git"
PINNED_COMMIT = "90f790542b4a5797e32a7c4951770524be79a021"
SRC_DIRS = ["content/en/docs/concepts", "content/en/docs/tasks"]
CLONE_DIR = REPO_ROOT / ".vendor-tmp" / "k8s-website"
OUT_DIR = REPO_ROOT / "corpus" / "kubernetes-docs"
MANIFEST_PATH = REPO_ROOT / "corpus" / "manifest.json"


def clone_pinned_commit():
    if CLONE_DIR.exists():
        return
    CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(CLONE_DIR)], check=True)
    subprocess.run(["git", "remote", "add", "origin", SOURCE_REPO], cwd=CLONE_DIR, check=True)
    subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", PINNED_COMMIT], cwd=CLONE_DIR, check=True
    )
    subprocess.run(["git", "checkout", "-q", PINNED_COMMIT], cwd=CLONE_DIR, check=True)
    subprocess.run(["git", "sparse-checkout", "init", "--cone"], cwd=CLONE_DIR, check=True)
    subprocess.run(["git", "sparse-checkout", "set", *SRC_DIRS], cwd=CLONE_DIR, check=True)
    subprocess.run(["git", "checkout", "-q", PINNED_COMMIT], cwd=CLONE_DIR, check=True)

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# {{< note >}} ... {{< /note >}}  and same for warning / caution
ADMONITION_RE = re.compile(
    r"\{\{<\s*(note|warning|caution)\s*>\}\}(.*?)\{\{<\s*/\1\s*>\}\}",
    re.DOTALL,
)
ADMONITION_LABEL = {"note": "Note", "warning": "Warning", "caution": "Caution"}

GLOSSARY_RE = re.compile(r"\{\{<\s*glossary_tooltip\s+([^>]*?)\s*>\}\}")
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')

# {{% heading "whatsnext" %}} etc. render to fixed section titles in the k8s docs i18n strings
HEADING_SHORTCODE_RE = re.compile(r'\{\{%\s*heading\s+"(\w+)"\s*%\}\}')
HEADING_SHORTCODE_TEXT = {
    "whatsnext": "What's next",
    "prerequisites": "Before you begin",
    "cleanup": "Cleaning up",
    "objectives": "Objectives",
    "seealso": "See also",
}

# Any other paired shortcode {{< tag ... >}} ... {{< /tag >}}: keep inner content, drop wrapper
GENERIC_PAIRED_RE = re.compile(r"\{\{<\s*(\w+)[^>]*>\}\}(.*?)\{\{<\s*/\1\s*>\}\}", re.DOTALL)

# Self-closing shortcodes: {{< tag ... >}} or {{% tag ... %}}
SELF_CLOSING_RE = re.compile(r"\{\{[<%][^}]*?[%>]\}\}")


def strip_admonitions(text: str) -> str:
    def repl(m):
        label = ADMONITION_LABEL[m.group(1)]
        body = m.group(2).strip()
        quoted = "\n".join(f"> {line}" if line else ">" for line in body.splitlines())
        return f"> **{label}:** \n{quoted}"

    return ADMONITION_RE.sub(repl, text)


def strip_glossary_tooltips(text: str) -> str:
    def repl(m):
        attrs = dict(ATTR_RE.findall(m.group(1)))
        if "text" in attrs:
            return attrs["text"]
        return attrs.get("term_id", "").replace("-", " ")

    return GLOSSARY_RE.sub(repl, text)


def strip_heading_shortcodes(text: str) -> str:
    return HEADING_SHORTCODE_RE.sub(lambda m: HEADING_SHORTCODE_TEXT[m.group(1)], text)


def strip_shortcodes(text: str) -> str:
    text = HTML_COMMENT_RE.sub("", text)
    text = strip_admonitions(text)
    text = strip_glossary_tooltips(text)
    text = strip_heading_shortcodes(text)
    # code_sample / feature-state / other self-closing/paired tags: drop, keep no placeholder noise
    text = GENERIC_PAIRED_RE.sub(lambda m: m.group(2), text)
    text = SELF_CLOSING_RE.sub("", text)
    return text


def extract_title(frontmatter: str, fallback: str) -> str:
    m = re.search(r'^title:\s*"?([^"\n]+)"?\s*$', frontmatter, re.MULTILINE)
    return m.group(1).strip() if m else fallback


def extract_description(frontmatter: str) -> str:
    # handles both `description: text` and the `description: >-\n  text` block-scalar form
    m = re.search(r"^description:\s*>?-?\s*\n((?:^ {1,}.+\n?)+)", frontmatter, re.MULTILINE)
    if m:
        return " ".join(line.strip() for line in m.group(1).strip().splitlines())
    m = re.search(r'^description:\s*"?([^"\n]+)"?\s*$', frontmatter, re.MULTILINE)
    return m.group(1).strip() if m else ""


def process_file(src: Path, rel_path: str) -> dict:
    raw = src.read_text(encoding="utf-8")
    fm_match = FRONTMATTER_RE.match(raw)
    frontmatter = fm_match.group(1) if fm_match else ""
    body = raw[fm_match.end():] if fm_match else raw

    title = extract_title(frontmatter, fallback=src.stem.replace("-", " ").title())
    description = extract_description(frontmatter)
    body = strip_shortcodes(body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"

    if not body.lstrip().startswith("# "):
        body = f"# {title}\n\n{body}"
    if description and description not in body:
        heading_line, rest = body.split("\n", 1)
        body = f"{heading_line}\n\n{description}\n{rest}"

    # extract headings (incl. explicit {#id} anchors) BEFORE stripping the {#id} markup for display
    headings = extract_headings(body)
    body = strip_explicit_ids_from_body(body)

    return {
        "rel_path": rel_path,
        "title": title,
        "body": body,
        "headings": headings,
    }


def main():
    clone_pinned_commit()

    commit_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=CLONE_DIR, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert commit_sha == PINNED_COMMIT, f"expected {PINNED_COMMIT}, got {commit_sha}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files_manifest = []

    for src_dir in SRC_DIRS:
        for src in sorted((CLONE_DIR / src_dir).rglob("*.md")):
            rel_to_docs = src.relative_to(CLONE_DIR / "content/en/docs")
            out_path = OUT_DIR / rel_to_docs
            out_path.parent.mkdir(parents=True, exist_ok=True)

            info = process_file(src, str(rel_to_docs))
            out_path.write_text(info["body"], encoding="utf-8")

            files_manifest.append(
                {
                    "path": info["rel_path"],
                    "title": info["title"],
                    "headings": info["headings"],
                }
            )

    manifest = {
        "source_repo": "https://github.com/kubernetes/website",
        "license": "CC BY 4.0",
        "commit_sha": commit_sha,
        "sections_included": SRC_DIRS,
        "file_count": len(files_manifest),
        "files": files_manifest,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"vendored {len(files_manifest)} files at commit {commit_sha}")


if __name__ == "__main__":
    main()
