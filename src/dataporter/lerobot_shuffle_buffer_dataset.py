"""LeRobot-compatible Dataset backed by a ShuffleBuffer.

Extends ShuffleBufferDataset to return complete LeRobot samples:
video frames from the ShuffleBuffer + non-video data (actions, states,
rewards, done flags) from the HuggingFace dataset via delta_timestamps
windowing.

Workers never decode video. The ProducerPool background process fills the
ShuffleBuffer, and workers only call ``sample()`` (shared-memory read)
plus fast HF dataset slicing for non-video columns.

**Growing-training-set support.** When HF-hosted datasets are prefetched
in the background, the training set must grow as new episodes land.
``refresh(min_new, timeout)`` re-scans the attached prefetchers, admits
newly-ready episodes that satisfy the split predicate, and forwards the
updated list to the ProducerPool.  ``__len__`` reflects current admitted
content.  See :class:`dataporter.GrowingDatasetCallback` for the Lightning
hook that drives refresh at epoch boundaries.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable

import torch
from torch.utils.data import Dataset

from .shuffle_buffer import ShuffleBuffer

logger = logging.getLogger(__name__)


def _default_train_split(ep_idx: int) -> bool:
    """Default stable split: 90% train, 10% val keyed by episode id.

    Episodes with ``ep_idx % 10 == 9`` go to val; everything else is
    train.  Stable across refreshes and dataset growth — an episode
    admitted in epoch 5 has the same train/val assignment as it would
    have had at epoch 1.
    """
    return ep_idx % 10 != 9


class LeRobotShuffleBufferDataset(Dataset):
    """LeRobot-compatible dataset backed by ShuffleBuffer.

    On each ``__getitem__`` call:
    1. Samples a random episode from the ShuffleBuffer (uint8 frames)
    2. Maps episode index to the correct source dataset
    3. Picks a random sample index within that episode
    4. Fetches non-video data via delta_timestamps windowing from HF dataset
    5. Extracts the corresponding frame window from the sampled frames
    6. Applies per-source transform if configured
    7. Returns a complete sample dict matching LeRobotDataset output format

    Args:
        buffer: ShuffleBuffer to sample from.
        sources: List of source dicts, each containing:
            - ``dataset``: FastLeRobotDataset instance
            - ``source_name``: string matching
              :attr:`ProducerConfig.source_name`, used by
              :meth:`refresh` when forwarding admitted lists to the pool
            - ``episode_offset``: shifts raw episode indices into a
              source-unique namespace
            - ``transform``: optional per-source transform callable
        delta_timestamps: Temporal windowing config (key -> list of delta times).
        prefetchers: Prefetchers to poll for ready episodes during
            :meth:`refresh`.  Usually one per source.  Pass an empty list
            for static datasets where the admitted set never changes.
        producer_pool: Optional ProducerPool that feeds the buffer.
            :meth:`refresh` forwards admitted lists via
            :meth:`ProducerPool.update_episodes`.
        split_fn: Predicate ``ep_idx -> bool`` selecting train episodes
            (True) vs val (False).  Defaults to ``ep_idx % 10 != 9``.
        default_min_new: Default ``min_new`` for :meth:`refresh` calls
            made without explicit arg.  0 (default) = non-blocking, admit
            what's ready right now.  Positive = block for that many new
            episodes each refresh (deterministic per-epoch cadence).
        image_keys: List of video/image keys in the dataset.
        seed: Random seed for per-worker RNG.
    """

    # Amortized stale-refresh detection tunables (instance-level overridable
    # for tests that need a faster trigger).
    _REFRESH_WARN_STALENESS_S: float = 120.0
    _REFRESH_WARN_MIN_GAP: int = 10
    _REFRESH_WARN_SAMPLE_PERIOD: int = 500

    def __init__(
        self,
        buffer: ShuffleBuffer,
        sources: list[dict],
        delta_timestamps: dict,
        prefetchers: list | None = None,
        producer_pool=None,
        split_fn: Callable[[int], bool] | None = None,
        default_min_new: int = 0,
        epoch_length: int | None = None,
        image_keys: list[str] | None = None,
        seed: int = 42,
    ):
        self._buffer = buffer
        self._sources = sources
        self._delta_timestamps = delta_timestamps
        self._prefetchers = list(prefetchers or [])
        self._producer_pool = producer_pool
        self._split_fn: Callable[[int], bool] = (
            split_fn if split_fn is not None else _default_train_split
        )
        self._default_min_new = int(default_min_new)
        self._image_keys = image_keys or ["observation.image"]
        self._seed = seed
        self._rng = random.Random(seed)

        # Per-source offsets are fixed at construction; the admitted
        # episode list rebuilds on every refresh().
        self._ep_offsets: list[int] = [
            src.get("episode_offset", 0) for src in self._sources
        ]
        self._source_names: list[str] = [
            src.get("source_name", f"source_{i}")
            for i, src in enumerate(self._sources)
        ]
        # Fully rebuilt by refresh().
        self._ep_to_source: dict[int, int] = {}
        self._current_train_episodes: list[int] = []
        self._epoch_length: int = 0

        # Back-compat: if the caller supplied ``train_episode_indices`` per
        # source (the pre-growing-set construction shape), admit it
        # immediately so ``__len__`` and sampling work without a
        # refresh() call.  Growing-mode callers pass prefetchers instead
        # and leave train_episode_indices off.
        initial: list[int] = []
        for src_idx, src in enumerate(self._sources):
            offset = self._ep_offsets[src_idx]
            for ep in src.get("train_episode_indices", []):
                initial.append(offset + ep)
        if initial:
            self._admit(sorted(set(initial)))

        # Legacy back-compat: an explicit ``epoch_length`` always wins
        # over whatever ``_admit`` computed, even when
        # ``train_episode_indices`` were supplied.  Pre-growing callers
        # used this knob to pin __len__ independent of the actual train
        # frame count; preserve that semantics.  Growing-mode callers
        # don't pass epoch_length, so refresh() drives the number.
        if epoch_length is not None:
            self._epoch_length = int(epoch_length)

        # Stale-refresh detection state.
        self._last_refresh_ts: float = time.monotonic()
        self._warned_stale: bool = False
        self._getitem_counter: int = 0

        # Pre-compute delta_indices (frame offsets from delta_timestamps).
        self._delta_indices: dict[str, list[int]] | None = None
        if delta_timestamps:
            fps = self._sources[0]["dataset"].fps
            self._delta_indices = {
                key: [round(d * fps) for d in deltas]
                for key, deltas in delta_timestamps.items()
            }
            self._fps = fps

    # ------------------------------------------------------------------
    # Growing-set API
    # ------------------------------------------------------------------

    def refresh(
        self,
        min_new: int | None = None,
        timeout: float | None = None,
    ) -> int:
        """Re-scan prefetchers, admit newly-ready train episodes.

        Args:
            min_new: Minimum number of *new* (previously-unadmitted) train
                episodes that must become ready before this call returns.
                ``None`` (default) → uses ``default_min_new`` from init.
                ``0`` → non-blocking, admit whatever's ready right now.
                Positive → poll until delta >= ``min_new`` OR every
                attached prefetcher is :meth:`done <Prefetcher.is_done>`
                (nothing more coming) OR timeout fires.
            timeout: Max seconds to wait for ``min_new``.  ``None`` =
                unlimited.  On expiry raises ``TimeoutError``.

        Returns:
            The new epoch length (total frames across admitted episodes).

        Raises:
            TimeoutError: ``min_new`` couldn't be satisfied in ``timeout``
                seconds.
        """
        effective_min_new = (
            self._default_min_new if min_new is None else int(min_new)
        )
        baseline = len(self._current_train_episodes)
        deadline: float | None = (
            time.monotonic() + timeout if timeout is not None else None
        )
        while True:
            ready_train = self._scan_ready_train_episodes()
            delta = len(ready_train) - baseline
            if delta >= effective_min_new:
                break
            if self._all_prefetchers_done():
                # Nothing more coming — admit what we have and return,
                # even if below min_new.  Bounded-run semantics.
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"refresh: {delta} new episodes ready after "
                    f"{timeout}s (needed {effective_min_new})"
                )
            time.sleep(0.1)

        self._admit(ready_train)
        self._last_refresh_ts = time.monotonic()
        return self._epoch_length

    def _scan_ready_train_episodes(self) -> list[int]:
        """All currently-ready episodes (across sources) that are train."""
        ready: list[int] = []
        for src_idx, pf in enumerate(self._prefetchers):
            offset = self._ep_offsets[src_idx] if src_idx < len(self._ep_offsets) else 0
            for raw_ep in pf.ready_episodes():
                if self._split_fn(raw_ep):
                    ready.append(offset + raw_ep)
        return sorted(set(ready))

    def _all_prefetchers_done(self) -> bool:
        if not self._prefetchers:
            return True
        return all(pf.is_done() for pf in self._prefetchers)

    def _admit(self, new_train: list[int]) -> None:
        """Install a new admitted list; no-op if unchanged."""
        if new_train == self._current_train_episodes:
            return
        self._current_train_episodes = list(new_train)

        # Rebuild ep → source mapping.  Each source_name has one offset;
        # raw_ep = keyed_ep - offset.
        self._ep_to_source = {}
        per_source: dict[str, list[int]] = {
            name: [] for name in self._source_names
        }
        for keyed_ep in new_train:
            src_idx = self._route_episode_to_source(keyed_ep)
            if src_idx is None:
                continue
            self._ep_to_source[keyed_ep] = src_idx
            offset = self._ep_offsets[src_idx]
            raw_ep = keyed_ep - offset
            per_source[self._source_names[src_idx]].append(raw_ep)

        # Recompute epoch length from per-source raw episode frame counts.
        total = 0
        for src_idx, src in enumerate(self._sources):
            offset = self._ep_offsets[src_idx]
            ep_data_index = src["dataset"].episode_data_index
            for keyed_ep in new_train:
                if self._ep_to_source.get(keyed_ep) != src_idx:
                    continue
                raw_ep = keyed_ep - offset
                total += int(
                    ep_data_index["to"][raw_ep]
                    - ep_data_index["from"][raw_ep]
                )
        self._epoch_length = total

        # Forward to the pool so new decodes sample the admitted set.
        if self._producer_pool is not None:
            for name, raw_list in per_source.items():
                self._producer_pool.update_episodes(name, raw_list)

    def _route_episode_to_source(self, keyed_ep: int) -> int | None:
        """Find which source owns this keyed (offset+raw) episode id.

        Relies on per-source offsets partitioning the id space.  For the
        common case of one source, always returns 0.
        """
        if len(self._sources) == 1:
            return 0
        # Pick the source whose offset gives a raw_ep within that
        # dataset's episode_data_index range.
        for src_idx, src in enumerate(self._sources):
            offset = self._ep_offsets[src_idx]
            raw_ep = keyed_ep - offset
            ep_data_index = src["dataset"].episode_data_index
            if 0 <= raw_ep < len(ep_data_index["from"]):
                return src_idx
        return None

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._epoch_length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        """Return a complete LeRobot sample from the buffer.

        ``idx`` is ignored -- every call samples uniformly from the buffer.
        Amortized stale-refresh check runs every
        :attr:`_REFRESH_WARN_SAMPLE_PERIOD` calls.
        """
        self._getitem_counter += 1
        if (
            self._getitem_counter % self._REFRESH_WARN_SAMPLE_PERIOD == 0
        ):
            self._maybe_warn_stale_refresh()

        # 1. Sample random episode from buffer
        # 2. Map to source dataset (retry up to 10 times on miss)
        for attempt in range(10):
            ep_idx, frames_uint8 = self._buffer.sample(self._rng)
            src_idx = self._ep_to_source.get(ep_idx)
            if src_idx is not None:
                break
        else:
            raise RuntimeError(
                f"No valid episode after 10 samples from buffer. "
                f"Known episodes: {sorted(self._ep_to_source.keys())[:10]}..."
            )

        source = self._sources[src_idx]
        dataset = source["dataset"]

        # 3. Pick random sample index within episode
        original_ep_idx = ep_idx - self._ep_offsets[src_idx]
        ep_data_index = dataset.episode_data_index
        ep_start = int(ep_data_index["from"][original_ep_idx])
        ep_end = int(ep_data_index["to"][original_ep_idx])
        num_frames_in_ep = ep_end - ep_start

        sample_idx = ep_start + self._rng.randint(0, num_frames_in_ep - 1)

        # 4. Fetch non-video data from HF dataset
        item = dataset.hf_dataset[sample_idx]

        if self._delta_indices is not None:
            query_indices = {}
            padding = {}
            for key, delta_idx in self._delta_indices.items():
                indices = [
                    max(ep_start, min(ep_end - 1, sample_idx + d))
                    for d in delta_idx
                ]
                query_indices[key] = indices
                padding[f"{key}_is_pad"] = torch.BoolTensor([
                    (sample_idx + d < ep_start) or (sample_idx + d >= ep_end)
                    for d in delta_idx
                ])

            item = {**item, **padding}

            query_result = dataset._query_hf_dataset(query_indices)
            for key, val in query_result.items():
                item[key] = val

        # 5. Extract video frame window from sampled frames
        frame_offset_in_ep = sample_idx - ep_start

        for vid_key in self._image_keys:
            if self._delta_indices is not None and vid_key in self._delta_indices:
                frame_indices = []
                for d in self._delta_indices[vid_key]:
                    abs_idx = max(ep_start, min(ep_end - 1, sample_idx + d))
                    rel_idx = abs_idx - ep_start
                    rel_idx = min(rel_idx, len(frames_uint8) - 1)
                    frame_indices.append(rel_idx)
                item[vid_key] = frames_uint8[frame_indices].to(torch.float32) / 255.0
            else:
                rel_idx = min(frame_offset_in_ep, len(frames_uint8) - 1)
                item[vid_key] = (
                    frames_uint8[rel_idx].unsqueeze(0).to(torch.float32) / 255.0
                )

        task_idx = item["task_index"].item() if hasattr(item["task_index"], "item") else int(item["task_index"])
        item["task"] = dataset.meta.tasks[task_idx]

        # 6. Apply per-source transform
        transform = source.get("transform")
        if transform is not None:
            item = transform(item)

        return item

    # ------------------------------------------------------------------
    # Stale-refresh detection
    # ------------------------------------------------------------------

    def _maybe_warn_stale_refresh(self) -> None:
        """Log once if unadmitted ready episodes sit while refresh() hasn't
        been called in a while.

        Catches the failure mode where a user wires a growing dataset but
        forgets to attach :class:`GrowingDatasetCallback` to their Trainer
        (or, in non-Lightning use, forgets to call ``refresh()``).  Silent
        otherwise.
        """
        if self._warned_stale:
            return
        if not self._prefetchers:
            return
        elapsed = time.monotonic() - self._last_refresh_ts
        if elapsed < self._REFRESH_WARN_STALENESS_S:
            return
        # How many train episodes are ready but not yet admitted?
        ready = self._scan_ready_train_episodes()
        gap = len(ready) - len(self._current_train_episodes)
        if gap < self._REFRESH_WARN_MIN_GAP:
            return
        logger.warning(
            f"LeRobotShuffleBufferDataset: {gap} train episodes have "
            f"become ready since the last refresh() {elapsed:.0f}s ago. "
            f"Training is currently iterating only "
            f"{len(self._current_train_episodes)} episodes — downloads "
            f"since then are not being used.  If using Lightning, add "
            f"GrowingDatasetCallback() to your Trainer.  Otherwise call "
            f"dataset.refresh() between epochs."
        )
        self._warned_stale = True

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        """Seed per-worker RNG for diverse sampling across workers.

        Must be passed as ``worker_init_fn`` to DataLoader to avoid
        correlated sampling across forked workers.
        """
        import torch as _torch

        info = _torch.utils.data.get_worker_info()
        if info is not None and hasattr(info.dataset, "_rng"):
            import random as _random
            info.dataset._rng = _random.Random(info.seed)
