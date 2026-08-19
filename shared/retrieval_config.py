"""Retrieval parameters locked once and held identical across all five
implementations, so differences in eval results reflect the orchestration
layer, not uneven retrieval setups.

Decided 2026-08-19 while building implementations/langchain_impl/ — the
first implementation to touch retrieval. See that implementation's
indexing.py for the chunking algorithm these parameters back.
"""

TOP_K = 5
SECTION_CHAR_LIMIT = 1000
CHUNK_OVERLAP = 150
