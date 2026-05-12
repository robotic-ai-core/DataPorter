"""Chat/instruction datasets with on-the-fly tokenization.

Two dataset classes sharing the same templating logic:

- **ChatDataset** (map-style): Tokenizes all examples on init and holds
  numpy arrays in RAM.  O(1) random access.  Best for small instruction
  datasets (10k–50k examples, ~50 MB in RAM).

- **ChatStreamDataset** (iterable): Sequential reads with a shuffle
  buffer and on-the-fly tokenization.  Best for large sharded datasets
  where preloading is impractical.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .template import apply_chat_template

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_tensors(row: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Convert numpy arrays from template to tensors for collation."""
    return {
        "input_ids": torch.as_tensor(row["input_ids"], dtype=torch.long),
        "labels": torch.as_tensor(row["labels"], dtype=torch.long),
        "loss_mask": torch.as_tensor(row["loss_mask"], dtype=torch.bool),
    }


# ------------------------------------------------------------------
# Map-style (preload into RAM)
# ------------------------------------------------------------------

class ChatDataset(Dataset):
    """Map-style chat/instruction dataset, preloaded in RAM.

    Loads a HuggingFace dataset (or any iterable of dicts), tokenizes
    every example with :func:`apply_chat_template`, and stores the
    resulting numpy arrays in a flat list.  Zero disk I/O after init.

    Args:
        source: HuggingFace dataset ID **or** a pre-loaded iterable of
            dicts (anything with text fields matching *field_map*).
        split: HuggingFace dataset split (ignored when *source* is an
            iterable).
        tokenizer: Tokenizer, already extended with special tokens.
        roles: ``{"query": "<|user|>", "response": "<|assistant|>",
            "end": "<|end|>"}``
        field_map: ``{"query": "prompt", "response": "completion"}``
        seq_len: Fixed sequence length.
        max_examples: Optional cap on number of source examples to
            process (useful for debugging).
    """

    def __init__(
        self,
        source: str | list[dict],
        split: str = "train",
        tokenizer=None,
        roles: dict[str, str] | None = None,
        field_map: dict[str, str] | None = None,
        seq_len: int = 512,
        max_examples: int | None = None,
    ):
        super().__init__()
        self._source_name = source if isinstance(source, str) else "custom"
        self._seq_len = seq_len
        self._tokenizer = tokenizer
        self._roles = roles
        self._field_map = field_map

        self._data = self._load_and_tokenize(source, split, max_examples)

        mb = self._data[0]["input_ids"].nbytes * len(self._data) / 1e6 if self._data else 0
        logger.info(
            f"ChatDataset: {len(self._data)} examples from "
            f"{self._source_name}/{split} "
            f"(seq_len={seq_len}, {mb:.1f} MB in RAM)"
        )

    # ---- loading ----

    def _load_and_tokenize(
        self,
        source: str | list[dict],
        split: str,
        max_examples: int | None,
    ) -> list[dict[str, np.ndarray]]:
        if isinstance(source, str):
            from datasets import load_dataset
            ds = load_dataset(source, split=split)
        else:
            ds = source

        results: list[dict[str, np.ndarray]] = []
        skipped = 0

        for i, example in enumerate(ds):
            if max_examples is not None and i >= max_examples:
                break
            result = apply_chat_template(
                example=example,
                tokenizer=self._tokenizer,
                roles=self._roles,
                field_map=self._field_map,
                seq_len=self._seq_len,
            )
            if result is not None:
                results.append(result)
            else:
                skipped += 1

        if skipped > 0:
            logger.info(
                f"Dropped {skipped} examples exceeding seq_len={self._seq_len}"
            )

        if not results:
            raise ValueError(
                f"No examples survived tokenization (all > seq_len={self._seq_len}). "
                f"Increase seq_len or check the dataset."
            )

        return results

    # ---- Dataset interface ----

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return _to_tensors(self._data[idx])

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def dataset_name(self) -> str:
        return self._source_name.split("/")[-1] if "/" in self._source_name else self._source_name


# ------------------------------------------------------------------
# Streaming (shuffle buffer, on-the-fly tokenization)
# ------------------------------------------------------------------

class ChatStreamDataset(IterableDataset):
    """Streaming chat/instruction dataset with shuffle buffer.

    Sequential reads through the source dataset, on-the-fly
    tokenization, and a fixed-size shuffle buffer for randomization
    without random seeks.

    **I/O pattern** (disk-friendly):

    1. Each worker gets a contiguous shard of the source data
       (modulo worker ID).
    2. Examples are tokenized on the fly and added to a RAM buffer
       (numpy arrays, ~200 bytes/example).
    3. When the buffer reaches *shuffle_buffer_size*, a random element
       is swapped out and yielded (O(1) swap-remove).
    4. At end of epoch, remaining buffer is shuffled and drained.

    Args:
        source: HuggingFace dataset ID.
        split: Dataset split.
        tokenizer: Tokenizer, already extended with special tokens.
        roles: Role token mapping.
        field_map: Field name mapping.
        seq_len: Fixed sequence length.
        shuffle_buffer_size: Number of tokenized examples to hold in
            the shuffle buffer.  Larger = better randomization, more RAM.
            10 000 examples ~ 20 MB at seq_len=512.
        seed: Base random seed (combined with epoch and worker ID).
    """

    def __init__(
        self,
        source: str,
        split: str = "train",
        tokenizer=None,
        roles: dict[str, str] | None = None,
        field_map: dict[str, str] | None = None,
        seq_len: int = 512,
        shuffle_buffer_size: int = 10_000,
        seed: int = 42,
    ):
        super().__init__()
        self._source = source
        self._split = split
        self._tokenizer = tokenizer
        self._roles = roles
        self._field_map = field_map
        self._seq_len = seq_len
        self._shuffle_buffer_size = shuffle_buffer_size
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic cross-worker shuffling."""
        self._epoch = epoch

    def __iter__(self):
        from datasets import load_dataset

        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        rng = random.Random(self._seed + self._epoch * 1000 + worker_id)

        # Streaming mode: HF datasets yields examples lazily from disk/network
        ds = load_dataset(self._source, split=self._split, streaming=True)

        buf: list[dict[str, np.ndarray]] = []

        for i, example in enumerate(ds):
            # Worker sharding: stripe examples across workers
            if i % num_workers != worker_id:
                continue

            result = apply_chat_template(
                example=example,
                tokenizer=self._tokenizer,
                roles=self._roles,
                field_map=self._field_map,
                seq_len=self._seq_len,
            )
            if result is None:
                continue

            buf.append(result)

            # Yield from buffer when full (O(1) swap-remove)
            while len(buf) >= self._shuffle_buffer_size:
                idx = rng.randrange(len(buf))
                row = buf[idx]
                buf[idx] = buf[-1]
                buf.pop()
                yield _to_tensors(row)

        # Drain remaining buffer in random order
        rng.shuffle(buf)
        for row in buf:
            yield _to_tensors(row)
