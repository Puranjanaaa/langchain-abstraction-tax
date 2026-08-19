import re

from slugify import slugify

HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
EXPLICIT_ID_RE = re.compile(r"\s*\{#([a-zA-Z0-9_.-]+)\}\s*$")


def strip_explicit_id(heading_text: str) -> tuple[str, str | None]:
    """Returns (clean_text, explicit_id_or_None)."""
    m = EXPLICIT_ID_RE.search(heading_text)
    if not m:
        return heading_text, None
    return heading_text[: m.start()].rstrip(), m.group(1)


def extract_headings(raw_body: str) -> list[dict]:
    """
    Parses heading lines from body text that may still contain `{#id}` attrs
    (i.e. call this BEFORE stripping them for display). Returns a list of
    {"level": int, "text": clean_text, "slug": slug} in document order, with
    auto-slug collisions disambiguated the way GitHub does (-1, -2, ...).
    """
    headings = []
    seen_slugs: dict[str, int] = {}

    for hashes, raw_text in HEADING_RE.findall(raw_body):
        clean_text, explicit_id = strip_explicit_id(raw_text)
        if explicit_id:
            slug = explicit_id
        else:
            base_slug = slugify(clean_text)
            count = seen_slugs.get(base_slug, 0)
            slug = base_slug if count == 0 else f"{base_slug}-{count}"
            seen_slugs[base_slug] = count + 1

        headings.append({"level": len(hashes), "text": clean_text, "slug": slug})

    return headings


def strip_explicit_ids_from_body(raw_body: str) -> str:
    """Removes trailing `{#id}` attrs from heading lines for clean display output."""

    def repl(m):
        hashes, text = m.group(1), m.group(2)
        clean_text, _ = strip_explicit_id(text)
        return f"{hashes} {clean_text}"

    return HEADING_RE.sub(repl, raw_body)
