"""Read-only Dataset backed by a ShuffleBuffer.

Workers call ``sample()`` to get a random item from whatever is currently
in the buffer. No index mapping, no cache misses, no video decode in workers.

Usage::

    buffer = ShuffleBuffer(capacity=800, ...)
    pool = ProducerPool(buffer, producers=[...])
    pool.start()
    pool.wait_for_warmup()

    dataset = ShuffleBufferDataset(buffer, epoch_length=5000)
    loader = DataLoader(dataset, batch_size=256, num_workers=4)
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from .shuffle_buffer import ShuffleBuffer


class ShuffleBufferDataset(Dataset):
    """Read-only dataset backed by ShuffleBuffer.

    Every ``__getitem__`` call returns a random sample from the buffer.
    The ``idx`` parameter is ignored — sampling is uniform over the
    buffer contents. This eliminates cache misses entirely.

    Args:
        buffer: The ShuffleBuffer to sample from.
        epoch_length: Synthetic dataset length (controls epoch frequency).
            Set to ``total_steps * batch_size`` or similar.
        seed: Random seed for per-worker RNG. Each worker gets a
            different seed via ``worker_init_fn``.
    """

    def __init__(
        self,
        buffer: ShuffleBuffer,
        epoch_length: int,
        seed: int = 42,
    ):
        self._buffer = buffer
        self._epoch_length = epoch_length
        self._seed = seed
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self._epoch_length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        """Return a random sample from the buffer.

        ``idx`` is ignored — every call samples uniformly from the
        buffer. Returns dict with 'frames' (uint8 tensor) and
        'episode_index' (int).
        """
        key, frames = self._buffer.sample(self._rng)
        return {"frames": frames, "episode_index": key}

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        """Seed per-worker RNG for diverse sampling across workers.

        Must be passed as ``worker_init_fn`` to DataLoader to avoid
        correlated sampling across forked workers (all inherit the
        same ``_rng`` state from the parent process).
        """
        info = torch.utils.data.get_worker_info()
        if info is not None and hasattr(info.dataset, "_rng"):
            info.dataset._rng = random.Random(info.seed)
