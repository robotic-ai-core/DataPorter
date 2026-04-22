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
            - ``shard_source``: :class:`LeRobotShardSource` instance
              (live, lazy read-only view of the dataset on disk).
            - ``source_name``: string matching
              :attr:`ProducerConfig.source_name`, used by
              :meth:`refresh` when forwarding admitted lists to the pool.
            - ``episode_offset``: shifts raw episode indices into a
              source-unique namespace (prevents key collisions across
              sources).
            - ``transform``: optional per-source transform callable.
            - ``train_episode_indices`` *(optional)*: raw ids for static
              (non-growing) datasets; admitted immediately at init.
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
        refresh_every_n_items: If set, the consumer calls
            :meth:`refresh` itself every N ``__getitem__`` invocations.
            Runs per-worker — each forked DataLoader worker self-refreshes
            independently, updating its local ``_ep_to_source`` and
            broadcasting the new episode list to the shared pool via its
            ``update_queue``.  With this set, neither
            :class:`GrowingDatasetCallback` nor
            ``reload_dataloaders_every_n_epochs=1`` is required: workers
            keep their admission maps fresh on their own.  ``None``
            (default) preserves the callback-driven flow.
        nominal_total_frames: If set, overrides ``__len__`` to a fixed
            value regardless of admission growth.  Used with
            ``refresh_every_n_items`` to give Lightning a stable
            ``num_training_batches`` so ``reload_dataloaders_every_n_epochs``
            and epoch-length-aware schedulers (OneCycleLR.total_steps)
            work correctly on streaming datasets.  ``None`` (default)
            preserves the live-growing ``_epoch_length`` semantics.
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
        refresh_every_n_items: int | None = None,
        nominal_total_frames: int | None = None,
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
        # Self-refresh knobs (see class docstring).  None preserves the
        # historical callback-driven flow.
        self._refresh_every_n_items: int | None = (
            int(refresh_every_n_items)
            if refresh_every_n_items is not None
            else None
        )
        self._nominal_total_frames: int | None = (
            int(nominal_total_frames)
            if nominal_total_frames is not None
            else None
        )

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

        # Static-set construction shape: callers that aren't using
        # prefetcher-driven growth pass ``train_episode_indices`` on each
        # source dict; admit them immediately so ``__len__`` and sampling
        # work without a refresh() call.  Growing-mode callers pass
        # prefetchers instead and leave train_episode_indices off.
        initial_by_source: dict[str, list[int]] = {}
        for src_idx, src in enumerate(self._sources):
            eps = list(src.get("train_episode_indices", []))
            if eps:
                initial_by_source[self._source_names[src_idx]] = eps
        if initial_by_source:
            self._admit_by_source(initial_by_source)

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

        # Source-name uniqueness is load-bearing for update_episodes()
        # routing — duplicates would silently broadcast to the wrong
        # producer.  Fail loud.
        if len(set(self._source_names)) != len(self._source_names):
            raise ValueError(
                f"LeRobotShuffleBufferDataset: duplicate source_name in "
                f"{self._source_names!r} — each source must have a unique "
                f"source_name (collision would break update_episodes routing)"
            )

        # Pre-compute delta_indices (frame offsets from delta_timestamps).
        self._delta_indices: dict[str, list[int]] | None = None
        if delta_timestamps:
            fps = int(self._sources[0]["shard_source"].fps)
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
            ready_by_source = self._scan_ready_train_episodes_by_source()
            total_ready = sum(len(v) for v in ready_by_source.values())
            delta = total_ready - baseline
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

        self._admit_by_source(ready_by_source)
        self._last_refresh_ts = time.monotonic()
        return self._epoch_length

    def _scan_ready_train_episodes_by_source(self) -> dict[str, list[int]]:
        """Per-source lists of currently-ready raw episode ids in train.

        Discovery goes through :meth:`LeRobotShardSource.list_ready_episodes`
        — the READER view of disk state — for every source.  This works
        uniformly for three cases:

        - Static (pre-populated local root, no prefetcher): the shard
          scans whatever is on disk right now.  refresh() is effectively
          idempotent in this case; admission doesn't change.
        - Dynamic (HF-prefetched): the prefetcher writes, the shard
          reads.  New episodes show up on the next scan.
        - Mixed (some of each): no special-casing — every source uses
          the same discovery path through its own shard.

        The prefetcher is a pure writer in this design.  Its own
        ``ready_episodes`` stays for internal min-ready gating but is
        no longer part of the consumer's read contract — see
        :class:`EpisodicPrefetcher` Protocol in ``interfaces.py``, which
        intentionally only requires ``is_done()`` for bounded-wait
        refresh semantics.
        """
        out: dict[str, list[int]] = {}
        for src_idx, src in enumerate(self._sources):
            name = self._source_names[src_idx]
            shard = src["shard_source"]
            ready = [
                raw for raw in shard.list_ready_episodes()
                if self._split_fn(raw)
            ]
            out[name] = sorted(set(ready))
        return out

    def _all_prefetchers_done(self) -> bool:
        if not self._prefetchers:
            return True
        return all(pf.is_done() for pf in self._prefetchers)

    def _admit_by_source(self, per_source: dict[str, list[int]]) -> None:
        """Install a new admitted set, keyed by source.

        Each entry maps ``source_name → raw_ids`` for that source.
        Source attribution is supplied by the caller so routing never
        depends on ``info.json``'s ``total_episodes`` (which can lie —
        stale downloads, in-flight updates, etc.).

        Defensive guard: an empty ``per_source`` dict that would clear a
        non-empty admitted set is treated as a no-op.  The scan path
        (:meth:`_scan_ready_train_episodes_by_source`) no longer produces
        this shape thanks to shard-driven discovery, but external
        callers may still hand us ``{}`` — don't wipe state on empty
        input.
        """
        if not per_source and self._current_train_episodes:
            logger.debug(
                "LeRobotShuffleBufferDataset._admit_by_source: empty "
                "per_source would wipe %d admitted episodes; treating as "
                "no-op",
                len(self._current_train_episodes),
            )
            return

        # Flat canonical form for idempotence check + sampling routing.
        flat: list[int] = []
        ep_to_source: dict[int, int] = {}
        per_source_normalized: dict[str, list[int]] = {
            name: [] for name in self._source_names
        }
        name_to_idx = {name: i for i, name in enumerate(self._source_names)}
        for name, raw_list in per_source.items():
            src_idx = name_to_idx.get(name)
            if src_idx is None:
                continue
            offset = self._ep_offsets[src_idx]
            raw_sorted = sorted(set(int(r) for r in raw_list))
            per_source_normalized[name] = raw_sorted
            for raw_ep in raw_sorted:
                keyed_ep = offset + raw_ep
                flat.append(keyed_ep)
                ep_to_source[keyed_ep] = src_idx
        flat.sort()

        if flat == self._current_train_episodes:
            return
        self._current_train_episodes = flat
        self._ep_to_source = ep_to_source

        # Epoch length = sum of per-source raw-episode frame counts.
        total = 0
        for name, raw_list in per_source_normalized.items():
            src_idx = name_to_idx[name]
            shard = self._sources[src_idx]["shard_source"]
            for raw_ep in raw_list:
                total += int(shard.episode_frame_count(raw_ep))
        self._epoch_length = total

        # Forward to the pool so new decodes sample the admitted set.
        if self._producer_pool is not None:
            for name, raw_list in per_source_normalized.items():
                self._producer_pool.update_episodes(name, raw_list)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        # When the caller pinned a nominal length at construction, honor
        # that regardless of admission growth — keeps Lightning's
        # num_training_batches stable and eliminates the need for
        # reload_dataloaders_every_n_epochs=1 on growing datasets.
        if self._nominal_total_frames is not None:
            return self._nominal_total_frames
        return self._epoch_length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        """Return a complete LeRobot sample from the buffer.

        ``idx`` is ignored -- every call samples uniformly from the buffer.
        Amortized stale-refresh check runs every
        :attr:`_REFRESH_WARN_SAMPLE_PERIOD` calls.  If
        ``refresh_every_n_items`` is set, a worker-local
        :meth:`refresh` fires every N calls — no external callback
        required.
        """
        self._getitem_counter += 1
        if (
            self._refresh_every_n_items
            and self._getitem_counter % self._refresh_every_n_items == 0
        ):
            # Worker-local refresh: rescans disk via prefetchers, updates
            # this worker's ``_ep_to_source``, and broadcasts the new
            # episode list to the shared producer pool via its
            # ``_update_queue`` (mp.Queue handles survive fork, so writes
            # from any worker reach the spawn-child pool).  Errors are
            # logged and swallowed — a transient disk scan failure
            # shouldn't crash the training step.
            try:
                self.refresh()
            except Exception as e:
                logger.warning(
                    f"LeRobotShuffleBufferDataset: self-refresh failed: "
                    f"{type(e).__name__}: {e}; continuing with current "
                    f"admitted set"
                )
        elif (
            self._getitem_counter % self._REFRESH_WARN_SAMPLE_PERIOD == 0
        ):
            # Stale-refresh warning only fires when self-refresh isn't
            # wired — otherwise the warning would be spurious.
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
        shard = source["shard_source"]

        # 3. Pick random sample index within episode
        raw_ep = ep_idx - self._ep_offsets[src_idx]
        num_frames_in_ep = int(shard.episode_frame_count(raw_ep))
        frame_in_ep = self._rng.randint(0, num_frames_in_ep - 1)

        # 4. Fetch non-video data (per-episode, local frame indices).
        item = shard.load_episode_row_torch(raw_ep, frame_in_ep)
        if self._delta_indices is not None:
            padding: dict[str, torch.Tensor] = {}
            for key, delta_idx in self._delta_indices.items():
                padding[f"{key}_is_pad"] = torch.BoolTensor([
                    (frame_in_ep + d < 0)
                    or (frame_in_ep + d >= num_frames_in_ep)
                    for d in delta_idx
                ])
                # Windowed non-video data pulled from the same
                # episode's parquet (skip video keys — filled from
                # frames_uint8 below).
                if key in self._image_keys:
                    continue
                local_indices = [
                    max(0, min(num_frames_in_ep - 1, frame_in_ep + d))
                    for d in delta_idx
                ]
                window = shard.load_episode_window_torch(
                    raw_ep, local_indices,
                )
                if key in window:
                    item[key] = window[key]
            item = {**item, **padding}

        # 5. Extract video frame window from sampled frames.  Indices are
        # LOCAL to the episode and clamped to the actually-decoded frame
        # count (episodes.jsonl can disagree with the mp4 at the margin).
        decoded_n = len(frames_uint8)
        for vid_key in self._image_keys:
            if self._delta_indices is not None and vid_key in self._delta_indices:
                frame_indices = [
                    min(
                        max(0, frame_in_ep + d),
                        min(num_frames_in_ep, decoded_n) - 1,
                    )
                    for d in self._delta_indices[vid_key]
                ]
                item[vid_key] = (
                    frames_uint8[frame_indices].to(torch.float32) / 255.0
                )
            else:
                rel_idx = min(frame_in_ep, decoded_n - 1)
                item[vid_key] = (
                    frames_uint8[rel_idx].unsqueeze(0).to(torch.float32) / 255.0
                )

        task_idx = (
            item["task_index"].item()
            if hasattr(item["task_index"], "item")
            else int(item["task_index"])
        )
        item["task"] = shard.tasks().get(task_idx, "")

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
        ready_by_source = self._scan_ready_train_episodes_by_source()
        total_ready = sum(len(v) for v in ready_by_source.values())
        gap = total_ready - len(self._current_train_episodes)
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
