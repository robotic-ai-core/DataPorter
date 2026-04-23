"""Shared-memory shuffle buffer for tokenized text data.

Variable-length-token analog of :class:`ShuffleBuffer`.  Pre-allocates
``[capacity, seq_len]`` token and mask tensors in shared memory; sequences
shorter than ``seq_len`` store their length in a companion ``lengths``
tensor and leave trailing slots as padding.

Single-writer design: only the :class:`TextProducerPool` child calls
``put()``.  DataLoader workers call ``sample()`` (read-only).  No locks
because each slot is written atomically by a single producer and workers
read fully-written slots only.

Usage::

    buffer = TokenShuffleBuffer(
        capacity=2000, seq_len=512, pad_token_id=0, vocab_size=8192,
    )
    # Producer process:
    buffer.put(doc_idx, tokens=..., loss_mask=...)
    # Worker process:
    key, tokens, loss_mask, length = buffer.sample(rng)
"""

from __future__ import annotations

import random

import torch

from ._rotation_gate import RotationGate


class TokenShuffleBuffer:
    """Shared-memory ring buffer for variable-length tokenized sequences.

    Layout (all in shared memory):
      - ``_tokens``       ``[capacity, seq_len]`` int32 — token ids, padded
      - ``_loss_mask``    ``[capacity, seq_len]`` uint8 — 1 where loss applies
      - ``_lengths``      ``[capacity]`` int32 — true sequence length (≤ seq_len)
      - ``_keys``         ``[capacity]`` int64 — slot identifier (for dedup/debug)
      - ``_write_head``   ``[1]`` int64 — total puts across the lifetime
      - ``_count``        ``[1]`` int64 — number of occupied slots

    Rotation is sample-gated via :class:`RotationGate` — same contract
    as :class:`ShuffleBuffer` on the video side, so both pipelines get
    identical rotation semantics from a single shared code path.  See
    ``_rotation_gate.py`` for the producer/consumer gate details.

    Args:
        capacity: Max number of items in the buffer.
        seq_len: Fixed sequence length (sequences are padded or truncated).
        pad_token_id: Token id used for padding past the true length.
        vocab_size: Optional vocab-size sanity check at ``put()`` time.
            When set, tokens ≥ vocab_size raise ValueError — catches
            tokenizer misconfiguration early rather than at embedding lookup.
        rotation_per_samples: K — samples consumed per producer put at
            steady state.  Default 1 = "rotate one slot per sample
            drawn" — natural for text because each slot holds exactly
            one document / training example, so K=1 means "each doc
            is served once before eviction" (maximum diversity, no
            repetition).  Set to ``None`` for direct-buffer tests or
            contexts with no producer pool present; the gate would
            otherwise block forever waiting for ``write_head`` to
            advance.  See :class:`RotationGate`.
    """

    def __init__(
        self,
        capacity: int,
        seq_len: int,
        pad_token_id: int = 0,
        vocab_size: int | None = None,
        rotation_per_samples: int | None = 1,
    ):
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")

        self._capacity = capacity
        self._seq_len = seq_len
        self._pad_token_id = int(pad_token_id)
        self._vocab_size = vocab_size
        self._gate = RotationGate(rotation_per_samples)

        # Pre-flight /dev/shm check
        token_bytes = capacity * seq_len * 4   # int32
        mask_bytes = capacity * seq_len * 1    # uint8
        overhead_bytes = (
            capacity * 4          # _lengths (int32)
            + capacity * 8        # _keys (int64)
            + 8                   # _write_head (int64)
            + 8                   # _count (int64)
        )
        total_bytes = token_bytes + mask_bytes + overhead_bytes
        self._check_shm_capacity(total_bytes)

        # Shared-memory tensors
        self._tokens = torch.full(
            (capacity, seq_len), self._pad_token_id, dtype=torch.int32
        ).share_memory_()
        self._loss_mask = torch.zeros(
            capacity, seq_len, dtype=torch.uint8
        ).share_memory_()
        self._lengths = torch.zeros(capacity, dtype=torch.int32).share_memory_()
        self._keys = torch.full((capacity,), -1, dtype=torch.int64).share_memory_()
        self._write_head = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._count = torch.zeros(1, dtype=torch.int64).share_memory_()

    @staticmethod
    def _check_shm_capacity(required_bytes: int) -> None:
        """Fail fast if /dev/shm can't hold the buffer.

        Docker defaults to 64 MB /dev/shm which silently breaks
        share_memory_() for large buffers.  Mirrors ShuffleBuffer's
        pre-flight — copy kept to avoid a cross-file dependency on a
        private helper.
        """
        import shutil
        from pathlib import Path

        shm = Path("/dev/shm")
        if not shm.exists():
            return  # non-Linux — skip

        usage = shutil.disk_usage(shm)
        required_gb = required_bytes / 1e9
        free_gb = usage.free / 1e9

        if required_bytes > usage.free:
            raise RuntimeError(
                f"TokenShuffleBuffer needs {required_gb:.2f} GB shared "
                f"memory but /dev/shm has only {free_gb:.2f} GB free. "
                f"Set --shm-size={max(2, int(required_gb * 1.5))}g in your "
                f"Docker run command."
            )

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def seq_len(self) -> int:
        return self._seq_len

    # ------------------------------------------------------------------
    # Back-compat accessors — the producer pool reads these directly.
    # Forwarded from :class:`RotationGate`; keeps the pool code
    # buffer-type-agnostic.
    # ------------------------------------------------------------------

    @property
    def _rotation_k(self) -> int | None:
        return self._gate.rotation_k

    @property
    def _samples_consumed(self):
        return self._gate._samples_consumed

    def __len__(self) -> int:
        return min(int(self._count), self._capacity)

    def __contains__(self, key: int) -> bool:
        return (self._keys == key).any().item()

    # ------------------------------------------------------------------
    # Writer API (single producer)
    # ------------------------------------------------------------------

    def put(
        self,
        key: int,
        tokens: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> int | None:
        """Write a tokenized sequence to the next slot.

        SINGLE WRITER ONLY — called exclusively by the producer.

        Args:
            key: Slot identifier (doc idx, example idx, etc).
            tokens: 1-D int tensor of tokens (any int dtype); truncated to
                ``seq_len`` if longer.  Shorter sequences are padded with
                ``pad_token_id``.
            loss_mask: 1-D uint8/bool tensor, same length as ``tokens``.
                When ``None`` all positions default to 1 (train on every
                token).

        Returns:
            The evicted key if this ``put`` overwrote an occupied slot,
            otherwise ``None``.
        """
        if tokens.dim() != 1:
            raise ValueError(
                f"tokens must be 1-D; got shape {tuple(tokens.shape)}"
            )

        length = min(int(tokens.shape[0]), self._seq_len)
        if length == 0:
            raise ValueError("tokens must have at least 1 element")

        token_slice = tokens[:length].to(torch.int32)

        if self._vocab_size is not None:
            max_id = int(token_slice.max())
            if max_id >= self._vocab_size:
                raise ValueError(
                    f"token id {max_id} >= vocab_size {self._vocab_size} "
                    "(tokenizer/buffer vocab mismatch)"
                )

        if loss_mask is None:
            mask_slice = torch.ones(length, dtype=torch.uint8)
        else:
            if loss_mask.dim() != 1:
                raise ValueError(
                    "loss_mask must be 1-D; got shape "
                    f"{tuple(loss_mask.shape)}"
                )
            if loss_mask.shape[0] != tokens.shape[0]:
                raise ValueError(
                    f"loss_mask length {loss_mask.shape[0]} != tokens "
                    f"length {tokens.shape[0]}"
                )
            mask_slice = loss_mask[:length].to(torch.uint8)

        slot = int(self._write_head) % self._capacity

        evicted = None
        old_key = int(self._keys[slot])
        if old_key >= 0 and int(self._count) >= self._capacity:
            evicted = old_key

        # Fill: real tokens + padding past the true length.  Writing the
        # tail is necessary because the slot may have stale content from
        # an earlier, longer sequence.
        self._tokens[slot, :length] = token_slice
        self._tokens[slot, length:] = self._pad_token_id
        self._loss_mask[slot, :length] = mask_slice
        self._loss_mask[slot, length:] = 0
        self._lengths[slot] = length
        self._keys[slot] = key

        # Publish AFTER the slot is fully written so readers never see a
        # half-filled slot.
        self._write_head[0] = int(self._write_head) + 1
        self._count[0] = min(int(self._count) + 1, self._capacity)

        return evicted

    # ------------------------------------------------------------------
    # Reader API (many DataLoader workers)
    # ------------------------------------------------------------------

    def sample(
        self,
        rng: random.Random,
    ) -> tuple[int, torch.Tensor, torch.Tensor, int]:
        """Return ``(key, tokens, loss_mask, length)`` for a random slot.

        ``tokens`` and ``loss_mask`` are **trimmed** to ``length`` — the
        worker never sees padding.  Callers that want fixed-shape batches
        should pad at collation time.

        Raises IndexError when the buffer is empty.
        """
        n = len(self)
        if n == 0:
            raise IndexError("TokenShuffleBuffer is empty")

        self._gate.wait_if_consumer_too_far_ahead(
            write_head_getter=lambda: int(self._write_head),
            capacity=self._capacity,
            buffer_name="TokenShuffleBuffer",
        )

        head = int(self._write_head)
        slot = (head - n + rng.randint(0, n - 1)) % self._capacity

        length = int(self._lengths[slot])
        key = int(self._keys[slot])
        tokens = self._tokens[slot, :length].clone()
        mask = self._loss_mask[slot, :length].clone()

        self._gate.record_sample()

        return key, tokens, mask, length

    def sample_padded(
        self,
        rng: random.Random,
    ) -> tuple[int, torch.Tensor, torch.Tensor, int]:
        """Like ``sample`` but returns padded ``[seq_len]`` tensors.

        Useful when the downstream collate_fn expects fixed shapes and
        callers want to skip per-sample padding.
        """
        n = len(self)
        if n == 0:
            raise IndexError("TokenShuffleBuffer is empty")

        self._gate.wait_if_consumer_too_far_ahead(
            write_head_getter=lambda: int(self._write_head),
            capacity=self._capacity,
            buffer_name="TokenShuffleBuffer",
        )

        head = int(self._write_head)
        slot = (head - n + rng.randint(0, n - 1)) % self._capacity

        length = int(self._lengths[slot])
        key = int(self._keys[slot])
        tokens = self._tokens[slot].clone()
        mask = self._loss_mask[slot].clone()

        self._gate.record_sample()

        return key, tokens, mask, length

    def keys(self) -> list[int]:
        """Return the list of keys currently occupying the buffer."""
        n = len(self)
        head = int(self._write_head)
        result = []
        for i in range(n):
            slot = (head - n + i) % self._capacity
            k = int(self._keys[slot])
            if k >= 0:
                result.append(k)
        return result

    def clear(self) -> None:
        self._tokens.fill_(self._pad_token_id)
        self._loss_mask.fill_(0)
        self._lengths.fill_(0)
        self._keys.fill_(-1)
        self._write_head.fill_(0)
        self._count.fill_(0)
        self._gate.reset()
