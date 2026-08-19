from dataclasses import dataclass, field


@dataclass(frozen=True)
class Citation:
    """A (file, heading) pair matching corpus/manifest.json's heading index.

    `file` is relative to corpus/kubernetes-docs/; `heading` is the slug from
    the manifest's per-file heading list — see scripts/headings.py.
    """

    file: str
    heading: str


@dataclass(frozen=True)
class Usage:
    """Token counts for a single answer() call, when the provider reports them."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    usage: Usage | None = None
