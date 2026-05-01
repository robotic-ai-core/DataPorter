"""Map-style Dataset backed by a TokenShuffleBuffer.

Workers call ``sample()`` on a shared-memory :class:`TokenShuffleBuffer`
filled by :class:`TextProducerPool`.  No tokenization happens in the
DataLoader worker — it's all been done in the producer child.

Usage::

    buffer = TokenShuffleBuffer(capacity=2000, seq_len=512, ...)
    pool = TextProducerPool(buffer, config=...)
    pool.start()
    pool.wait_for_warmup()

    dataset = TokenShuffleBufferDataset(buffer, epoch_length=5000)
    loader = DataLoader(
        dataset, batch_size=32, num_workers=4,
        worker_init_fn=TokenShuffleBufferDataset.worker_init_fn,
    )
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from .token_shuffle_buffer import TokenShuffleBuffer


class TokenShuffleBufferDataset(Dataset):
    """Dataset that samples tokenized sequences from a shared-memory buffer.

    Args:
        buffer: Filled by a :class:`TextProducerPool` before workers start.
        epoch_length: Virtual length reported to Lightning / samplers.
            The buffer is a rotating window — the true "dataset size" is
            not stable — so callers pick an epoch length that reflects the
            training cadence they want.
        padded: When True (default) returns fixed-shape ``[seq_len]``
            padded tensors so the default collate_fn works without a
            custom pad-and-stack step.  When False returns variable-length
            tensors trimmed to true length — useful if the downstream
            collate does its own packing.
        seed: Base RNG seed.  Per-worker seed is derived in
            ``worker_init_fn`` from ``get_worker_info().seed``.
    """

    def __init__(
        self,
        buffer: TokenShuffleBuffer,
        epoch_length: int,
        padded: bool = True,
        seed: int = 42,
    ):
        if epoch_length < 1:
            raise ValueError(f"epoch_length must be >= 1, got {epoch_length}")
        self._buffer = buffer
        self._epoch_length = epoch_length
        self._padded = padded
        self._seed = seed
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self._epoch_length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # idx is positional — the buffer is a rotating window, so we
        # always sample randomly from whatever is currently resident.
        if self._padded:
            key, tokens, mask, length = self._buffer.sample_padded(self._rng)
        else:
            key, tokens, mask, length = self._buffer.sample(self._rng)

        return {
            # Models (HF, torch.nn.Embedding) expect int64 input_ids.
            # Buffer stores int32 to halve the shared-memory footprint;
            # cast on the worker side.
            "input_ids": tokens.long(),
            "loss_mask": mask,
            "length": torch.tensor(length, dtype=torch.int32),
            "key": torch.tensor(key, dtype=torch.int64),
        }

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        """Seed each worker's RNG from its unique DataLoader seed.

        DataLoader assigns ``info.seed = base_seed + worker_id``, so
        workers produce uncorrelated sample streams.  Without this,
        every forked worker inherits the same RNG state and samples
        identical buffer slots.
        """
        info = torch.utils.data.get_worker_info()
        if info is None:
            return
        ds = info.dataset
        if hasattr(ds, "_rng"):
            ds._rng = random.Random(info.seed)
