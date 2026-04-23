"""Single source of truth for LeRobot sample construction.

Before this module, sample-building lived in two places:

- :meth:`LeRobotShuffleBufferDataset.__getitem__` — for the streaming
  train path, where the producer pool has already decoded a full
  episode into uint8 frames sitting in the :class:`ShuffleBuffer`.
- :meth:`FastLeRobotDataset.__getitem__` (via its overrides of
  ``_query_hf_dataset`` / ``_query_videos``) — for the map-style val
  path, backed by the HuggingFace Arrow cache and on-demand video
  decoding.

Both implementations do the same thing at their core: given
``(raw_episode_id, frame_within_episode)`` plus a ``delta_timestamps``
window spec, build a ``dict[str, Tensor]`` sample matching the LeRobot
dataset output contract.  They diverge only in where the row data
comes from (parquet vs. Arrow cache) and whether the video frames are
pre-decoded (train) or decoded on-demand (val).

:class:`SampleReader` is that shared primitive.  Given a
:class:`LeRobotShardSource` + windowing config, it exposes one method
— :meth:`read` — that returns the sample.  Callers supply pre-decoded
frames when they have them (buffer-backed train) or let the reader
decode lazily (map-style val, with a small per-episode LRU).

Extracting the primitive also fixes a drift risk: the two prior
implementations had subtle differences around clamping,
padding-mask computation, and task lookup that crept in as each got
patched separately.  One implementation, one contract.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from pathlib import Path

    from .lerobot_shard_source import LeRobotShardSource

logger = logging.getLogger(__name__)


def decode_episode_frames_uint8(
    video_path: "Path",
    num_frames: int,
    fps: int,
    *,
    tolerance_s: float | None = None,
    video_backend: str = "pyav",
) -> torch.Tensor:
    """Decode all frames of an episode into a ``[T, C, H, W]`` uint8 tensor.

    Mirrors :func:`dataporter.fast_lerobot_dataset.decode_episode_frames`
    but lives here so the val path doesn't depend on
    :mod:`fast_lerobot_dataset` (which will become deprecated for
    BlendedLeRobotDataModule's internal setup path).

    Args:
        video_path: Absolute path to the episode's video file.
        num_frames: Episode frame count (from ``episodes.jsonl``).
        fps: Episode frames per second.
        tolerance_s: Timestamp tolerance forwarded to LeRobot's video
            decoder.  ``None`` (default) → ``1 / fps / 4`` — matches
            :class:`LeRobotDataset`'s internal default.
        video_backend: Video backend passed to LeRobot's decoder.
    """
    # Local import so mock patches at the canonical location take effect
    # (matches the indirection used in :mod:`fast_lerobot_dataset`).
    from lerobot.common.datasets.video_utils import (
        decode_video_frames as _decode,
    )
    if tolerance_s is None:
        tolerance_s = 1.0 / float(fps) / 4.0
    all_ts = [i / float(fps) for i in range(num_frames)]
    frames = _decode(video_path, all_ts, tolerance_s, video_backend)
    if frames.dim() == 5:
        frames = frames.squeeze(0)
    return (frames * 255).to(torch.uint8)


class SampleReader:
    """Build LeRobot-compatible samples from ``(raw_ep, frame_in_ep)``.

    Callers supply pre-decoded uint8 frames when they have them (train
    path, from :class:`ShuffleBuffer`), or let the reader decode
    on-demand with an internal per-episode LRU (val path).

    Args:
        shard_source: Lazy read-only view of the on-disk LeRobot
            dataset.  Supplies fps, ``episode_frame_count``, row
            access, and the video path template.
        delta_timestamps: Optional ``{key: [delta_s, ...]}`` — the
            standard LeRobot temporal windowing spec.  ``None``
            (default) means no windowing: each sample is a single
            frame.
        image_keys: Video/image column names.  Defaults to
            ``["observation.image"]`` to match the prior consumer
            default.
        decode_cache_maxsize: LRU size for on-demand episode decodes.
            Only consulted when ``read()`` is called with
            ``frames_uint8=None``.  Default 4 — small enough to not
            hoard memory, large enough that consecutive frames within
            an episode hit the cache.
        video_backend: Forwarded to
            :func:`decode_episode_frames_uint8` on the on-demand path.

    Thread-safety: the decode LRU is unguarded.  One reader per
    DataLoader worker is the expected usage pattern; don't share a
    reader across threads.
    """

    def __init__(
        self,
        shard_source: "LeRobotShardSource",
        *,
        delta_timestamps: dict[str, list[float]] | None = None,
        image_keys: list[str] | None = None,
        decode_cache_maxsize: int = 4,
        video_backend: str = "pyav",
    ) -> None:
        self._shard = shard_source
        self._image_keys = list(image_keys) if image_keys else [
            "observation.image",
        ]
        self._delta_timestamps = delta_timestamps

        # Precompute frame-offset deltas once.  Matches
        # ``lerobot.common.datasets.utils.get_delta_indices`` — single
        # source of truth for "which frames do we need for this delta
        # spec at this fps".
        self._delta_indices: dict[str, list[int]] | None = None
        if delta_timestamps:
            fps = int(shard_source.fps)
            self._delta_indices = {
                key: [round(d * fps) for d in deltas]
                for key, deltas in delta_timestamps.items()
            }

        self._video_backend = video_backend
        self._decode_cache_maxsize = int(decode_cache_maxsize)
        # Populated lazily on first on-demand decode.
        self._decode_cache: "OrderedDict[int, torch.Tensor] | None" = None

        # Tasks mapping is small (~dozens of entries) and stable.
        # Cache it once per reader to avoid re-parsing ``tasks.jsonl``.
        self._tasks_cache: dict[int, str] | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def image_keys(self) -> list[str]:
        return list(self._image_keys)

    @property
    def delta_indices(self) -> dict[str, list[int]] | None:
        return None if self._delta_indices is None else {
            k: list(v) for k, v in self._delta_indices.items()
        }

    def read(
        self,
        raw_ep: int,
        frame_in_ep: int,
        frames_uint8: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Return a complete LeRobot sample at ``(raw_ep, frame_in_ep)``.

        Args:
            raw_ep: Raw episode id (matches ``episode_index`` in the
                underlying parquet).
            frame_in_ep: Local frame index within the episode, in
                ``[0, episode_frame_count)``.
            frames_uint8: Optional pre-decoded ``[T, C, H, W]`` uint8
                tensor for this episode.  Supplied by the train path
                (from :class:`ShuffleBuffer`).  Omit for the val path;
                the reader decodes on-demand with an internal LRU.
        """
        num_frames_in_ep = int(self._shard.episode_frame_count(raw_ep))

        # Non-video row data.
        item = self._shard.load_episode_row_torch(raw_ep, frame_in_ep)

        # Windowed non-video fields + per-key padding masks.
        if self._delta_indices is not None:
            item = self._apply_delta_windows(
                item, raw_ep, frame_in_ep, num_frames_in_ep,
            )

        # Videos.
        if frames_uint8 is None:
            frames_uint8 = self._decode_cached(raw_ep, num_frames_in_ep)
        item = self._apply_video_windows(
            item, frame_in_ep, num_frames_in_ep, frames_uint8,
        )

        # Task string lookup.  The row has an integer ``task_index``;
        # the consumer's sample contract includes the human-readable
        # task string alongside it.
        task_val = item["task_index"]
        task_idx = int(
            task_val.item() if hasattr(task_val, "item") else task_val
        )
        item["task"] = self._tasks().get(task_idx, "")

        return item

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tasks(self) -> dict[int, str]:
        if self._tasks_cache is None:
            self._tasks_cache = self._shard.tasks()
        return self._tasks_cache

    def _apply_delta_windows(
        self,
        item: dict[str, Any],
        raw_ep: int,
        frame_in_ep: int,
        num_frames_in_ep: int,
    ) -> dict[str, Any]:
        """Attach windowed non-video fields and ``*_is_pad`` flags.

        The padding flag reflects whether ``frame_in_ep + delta`` would
        fall outside the episode — independent of how we clamp the
        actual lookup index (which always stays in-bounds to avoid row
        errors).  This matches :class:`LeRobotDataset`'s contract: the
        caller sees "yes, this slot is padding" via the flag, and a
        clamped (duplicate-boundary) value otherwise.
        """
        assert self._delta_indices is not None
        padding: dict[str, torch.Tensor] = {}
        for key, delta_idx in self._delta_indices.items():
            padding[f"{key}_is_pad"] = torch.BoolTensor([
                (frame_in_ep + d < 0)
                or (frame_in_ep + d >= num_frames_in_ep)
                for d in delta_idx
            ])
            # Images are served separately from pre-decoded or
            # on-demand frames; skip the row-side windowed read for
            # them.
            if key in self._image_keys:
                continue
            local_indices = [
                max(0, min(num_frames_in_ep - 1, frame_in_ep + d))
                for d in delta_idx
            ]
            window = self._shard.load_episode_window_torch(
                raw_ep, local_indices,
            )
            if key in window:
                item[key] = window[key]
        return {**item, **padding}

    def _apply_video_windows(
        self,
        item: dict[str, Any],
        frame_in_ep: int,
        num_frames_in_ep: int,
        frames_uint8: torch.Tensor,
    ) -> dict[str, Any]:
        """Attach per-video-key frame tensors from ``frames_uint8``.

        ``frames_uint8`` may be shorter than ``num_frames_in_ep`` if
        the source's ``episodes.jsonl`` slightly disagrees with the
        actual mp4 frame count at the margin.  Clamp the local window
        indices against ``min(num_frames_in_ep, decoded_n)`` so we
        never index past either source.
        """
        decoded_n = int(len(frames_uint8))
        for vid_key in self._image_keys:
            if (
                self._delta_indices is not None
                and vid_key in self._delta_indices
            ):
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
                    frames_uint8[rel_idx]
                    .unsqueeze(0)
                    .to(torch.float32)
                    / 255.0
                )
        return item

    def _decode_cached(
        self, raw_ep: int, num_frames: int,
    ) -> torch.Tensor:
        """Decode a full episode to uint8 frames; LRU-cache the result.

        Only consulted on the val path (``frames_uint8=None`` in
        :meth:`read`).  Sized to ``decode_cache_maxsize`` so
        consecutive frames within one episode hit the cache.
        """
        if self._decode_cache is None:
            self._decode_cache = OrderedDict()
        if raw_ep in self._decode_cache:
            self._decode_cache.move_to_end(raw_ep)
            return self._decode_cache[raw_ep]

        vkeys = self._shard.video_keys
        if not vkeys:
            raise RuntimeError(
                f"SampleReader: shard at {self._shard.root!r} has no video "
                f"keys but a video-bearing sample was requested.  Pass "
                f"frames_uint8 explicitly or configure the shard with videos."
            )
        # Use the first image_key that's actually a video column, else
        # fall back to the shard's first video key.  Matches the
        # existing consumer's behavior.
        vk = next(
            (k for k in self._image_keys if k in vkeys), vkeys[0],
        )
        video_path = self._shard.episode_video_path(raw_ep, vk)
        frames = decode_episode_frames_uint8(
            video_path, num_frames, int(self._shard.fps),
            video_backend=self._video_backend,
        )

        self._decode_cache[raw_ep] = frames
        while len(self._decode_cache) > self._decode_cache_maxsize:
            self._decode_cache.popitem(last=False)
        return frames
