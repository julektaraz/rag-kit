import pytest

from ingestion.chunking import chunk_by_sentences


def test_chunking_basic_overlap():
    text = (
        "This is sentence one. "
        "This is sentence two. "
        "This is sentence three. "
        "This is sentence four."
    )

    chunks = chunk_by_sentences(
        text,
        sentences_per_chunk=2,
        overlap_sentences=1,
        min_chunk_size=10,
    )

    # We should get at least two chunks with some overlap in content.
    assert len(chunks) >= 2
    assert "sentence one" in chunks[0]
    assert "sentence two" in chunks[0]
    # Overlap means sentence two should also appear at the start of the next chunk.
    assert any("sentence two" in c for c in chunks[1:])


def test_chunking_empty_text_returns_empty_list():
    assert chunk_by_sentences("", min_chunk_size=10) == []


def test_chunking_short_text_single_chunk():
    text = "Short but meaningful sentence."
    chunks = chunk_by_sentences(text, min_chunk_size=5)
    assert chunks == [text]


