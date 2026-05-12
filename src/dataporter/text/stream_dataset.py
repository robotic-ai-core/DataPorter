"""Streaming pre-training dataset with on-the-fly tokenization.

Streams documents from a HuggingFace dataset, tokenizes and chunks
them into fixed-length sequences on the fly, and serves them through
a shuffle buffer.  Training starts as soon as the buffer fills —
no blocking pre-download step.

Optimizations:
  - Raw ``tokenizers`` library (no HF wrapper overhead)
  - ``encode_batch()`` for ~4x faster tokenization
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .chunking import TokenChunker

logger = logging.getLogger(__name__)


def _to_tensors(chunk: np.ndarray) -> dict[str, torch.Tensor]:
    """Convert a uint16 chunk to the standard {input_ids, labels} format."""
    ids = torch.as_tensor(chunk, dtype=torch.long)
    return {"input_ids": ids, "labels": ids.clone()}


def _load_raw_tokenizer(tokenizer_name: str):
    """Load a raw ``tokenizers.Tokenizer`` for fast encode_batch.

    Falls back to HF AutoTokenizer if the raw file isn't available.
    """
    from pathlib import Path

    path = Path(tokenizer_name) / "tokenizer.json"
    if path.exists():
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(path))

    # Fallback: HF tokenizer (slower but always works)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name)


def _is_raw_tokenizer(tok) -> bool:
    """Check if tok is a raw tokenizers.Tokenizer."""
    return hasattr(tok, "encode_batch") and not hasattr(tok, "vocab_size")


def _encode_texts(tok, is_raw: bool, texts: list[str]) -> list[list[int]]:
    """Batch-encode texts using the fastest available method."""
    if is_raw:
        return [enc.ids for enc in tok.encode_batch(texts)]
    return [tok.encode(text) for text in texts]


class PretrainStreamDataset(IterableDataset):
    """Streaming pre-training dataset with shuffle buffer.

    Streams documents from HuggingFace, tokenizes on the fly, chunks
    into ``seq_len`` sequences, and serves through a shuffle buffer.
    Training starts as soon as the buffer fills — no blocking download.

    Args:
        dataset: HuggingFace dataset ID (e.g. "HuggingFaceTB/smollm-corpus").
        data_dir: Subdirectory within the dataset (e.g. "fineweb-edu-dedup").
        split: HF dataset split (default: "train").
        text_field: Name of the text column.
        tokenizer_name: Tokenizer name or local path.
        seq_len: Fixed sequence length for chunks.
        shuffle_buffer_size: Number of chunks held in the shuffle buffer.
        max_docs: Maximum documents to stream (None = unlimited).
        tokenizer_batch_size: Documents per batch-tokenize call.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: str,
        data_dir: str | None = None,
        split: str = "train",
        text_field: str = "text",
        tokenizer_name: str = "gpt2",
        seq_len: int = 2048,
        shuffle_buffer_size: int = 10_000,
        max_docs: int | None = None,
        tokenizer_batch_size: int = 64,
        seed: int = 42,
    ):
        super().__init__()
        self._dataset = dataset
        self._data_dir = data_dir
        self._split = split
        self._text_field = text_field
        self._tokenizer_name = tokenizer_name
        self._seq_len = seq_len
        self._shuffle_buffer_size = shuffle_buffer_size
        self._max_docs = max_docs
        self._tokenizer_batch_size = tokenizer_batch_size
        self._seed = seed
        self._epoch = 0

        # Pre-load tokenizer to cache files and get EOT id
        tok = _load_raw_tokenizer(tokenizer_name)
        if _is_raw_tokenizer(tok):
            self._eot_id = tok.token_to_id("<|endoftext|>") or 0
        else:
            self._eot_id = tok.eos_token_id

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic cross-worker shuffling."""
        self._epoch = epoch

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def __iter__(self):
        from datasets import load_dataset

        rng = random.Random(self._seed + self._epoch * 1000)
        tok = _load_raw_tokenizer(self._tokenizer_name)
        is_raw = _is_raw_tokenizer(tok)
        chunker = TokenChunker(seq_len=self._seq_len, eot_token_id=self._eot_id)

        ds = load_dataset(
            self._dataset,
            data_dir=self._data_dir,
            split=self._split,
            streaming=True,
        )

        buf: list[np.ndarray] = []
        total_docs = 0
        batch_texts: list[str] = []

        for i, doc in enumerate(ds):
            text = doc.get(self._text_field, "")
            if not text.strip():
                continue

            batch_texts.append(text)

            if len(batch_texts) >= self._tokenizer_batch_size:
                for token_ids in _encode_texts(tok, is_raw, batch_texts):
                    for chunk in chunker.add_document(token_ids):
                        buf.append(chunk)

                        while len(buf) >= self._shuffle_buffer_size:
                            idx = rng.randrange(len(buf))
                            row = buf[idx]
                            buf[idx] = buf[-1]
                            buf.pop()
                            yield _to_tensors(row)

                total_docs += len(batch_texts)
                batch_texts.clear()

            if self._max_docs and (i + 1) >= self._max_docs:
                break

        # Flush remaining batch
        if batch_texts:
            for token_ids in _encode_texts(tok, is_raw, batch_texts):
                for chunk in chunker.add_document(token_ids):
                    buf.append(chunk)

                    while len(buf) >= self._shuffle_buffer_size:
                        idx = rng.randrange(len(buf))
                        row = buf[idx]
                        buf[idx] = buf[-1]
                        buf.pop()
                        yield _to_tensors(row)

            total_docs += len(batch_texts)

        # Flush chunker
        for chunk in chunker.flush():
            buf.append(chunk)

        # Drain remaining buffer
        rng.shuffle(buf)
        for row in buf:
            yield _to_tensors(row)

        logger.info(f"Streamed {total_docs:,} docs")
