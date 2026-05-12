"""Unit tests for TokenChunker."""

from __future__ import annotations

import numpy as np

from dataporter.text import TokenChunker


SEQ_LEN = 8
EOT = 0


def test_empty_document_returns_no_chunks():
    c = TokenChunker(seq_len=SEQ_LEN, eot_token_id=EOT)
    assert c.add_document([]) == []


def test_document_shorter_than_seq_len_yields_nothing():
    c = TokenChunker(seq_len=SEQ_LEN, eot_token_id=EOT)
    chunks = c.add_document([1, 2, 3])  # 3 tokens + EOT = 4, less than 8
    assert chunks == []
    assert c.buffer_size == 4


def test_document_emits_chunk_at_seq_len_boundary():
    c = TokenChunker(seq_len=SEQ_LEN, eot_token_id=EOT)
    chunks = c.add_document([1, 2, 3, 4, 5, 6, 7])  # 7 tokens + EOT = 8 → one chunk
    assert len(chunks) == 1
    assert chunks[0].dtype == np.uint16
    assert chunks[0].tolist() == [1, 2, 3, 4, 5, 6, 7, EOT]


def test_document_emits_multiple_chunks():
    c = TokenChunker(seq_len=SEQ_LEN, eot_token_id=EOT)
    chunks = c.add_document([1] * 20)  # 20 tokens + EOT = 21 → 2 chunks of 8, 5 carry
    assert len(chunks) == 2
    assert all(len(ch) == SEQ_LEN for ch in chunks)
    assert c.buffer_size == 5


def test_carry_across_documents():
    c = TokenChunker(seq_len=SEQ_LEN, eot_token_id=EOT)
    c.add_document([1, 2, 3])  # buffer: [1,2,3,EOT] (4)
    chunks = c.add_document([4, 5, 6, 7])  # buffer: [1,2,3,EOT,4,5,6,7,EOT] (9)
    # First 8 emitted; 1 left in buffer
    assert len(chunks) == 1
    assert chunks[0].tolist() == [1, 2, 3, EOT, 4, 5, 6, 7]
    assert c.buffer_size == 1


def test_flush_pads_remaining_with_eot():
    c = TokenChunker(seq_len=SEQ_LEN, eot_token_id=EOT)
    c.add_document([1, 2, 3])  # 4 tokens in buffer
    chunks = c.flush()
    assert len(chunks) == 1
    assert chunks[0].tolist() == [1, 2, 3, EOT, EOT, EOT, EOT, EOT]
    assert c.buffer_size == 0


def test_flush_empty_buffer_returns_empty():
    c = TokenChunker(seq_len=SEQ_LEN, eot_token_id=EOT)
    assert c.flush() == []


def test_reset_clears_buffer():
    c = TokenChunker(seq_len=SEQ_LEN, eot_token_id=EOT)
    c.add_document([1, 2, 3])
    c.reset()
    assert c.buffer_size == 0
    assert c.flush() == []
