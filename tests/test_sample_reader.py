"""Unit tests for :class:`SampleReader` — the shared sample-construction
primitive.

Covers:

- Shape/dtype contract of ``read()`` output, with and without
  ``delta_timestamps``.
- Supplied-frames path (train-side caller) vs on-demand decode path
  (val-side caller with the internal LRU).
- Padding masks reflect deltas that leave the episode, even though
  the looked-up values themselves are clamped in-bounds.
- Task-string lookup matches the shard's ``tasks.jsonl``.
- Decode LRU evicts correctly.
- Cross-path equivalence: SampleReader output at ``(ep, frame)`` is
  identical whether frames are supplied pre-decoded or decoded
  lazily.  This is the load-bearing invariant for the train/val
  split — both paths MUST agree at the byte level.

The tests reuse ``_make_dataset`` from
``test_shard_source_pool_e2e`` so the synthetic fixture is the same
one the end-to-end tests exercise.
"""

from __future__ import annotations

import pytest

pytest.importorskip("lerobot")
pytest.importorskip("imageio")
pytest.importorskip("imageio_ffmpeg")

import torch

from dataporter.lerobot_shard_source import LeRobotShardSource
from dataporter.sample_reader import SampleReader
from test_shard_source_pool_e2e import _make_dataset


# ---------------------------------------------------------------------------
# Shape/dtype contract
# ---------------------------------------------------------------------------


class TestBasicReadContract:

    def test_read_returns_expected_keys_without_delta(self, tmp_path):
        """Without ``delta_timestamps``, ``read()`` returns the episode
        row + a single-frame video tensor."""
        root = tmp_path / "ds"
        _make_dataset(root, ready_eps=[0], total_episodes=1)
        src = LeRobotShardSource(root)

        reader = SampleReader(src)
        item = reader.read(raw_ep=0, frame_in_ep=5)

        # Row fields present (episode_index/frame_index/task_index come
        # from the fixture's parquet; task comes from the reader).
        assert "episode_index" in item
        assert "frame_index" in item
        assert "task_index" in item
        assert "task" in item
        # Video: default image_key is ``observation.image``; single-frame
        # path unsqueezes to [1, C, H, W].
        assert "observation.image" in item
        frames = item["observation.image"]
        assert frames.ndim == 4
        assert frames.shape[0] == 1
        assert frames.dtype == torch.float32

    def test_read_with_delta_returns_windowed_video(self, tmp_path):
        """With delta_timestamps on the image key, the video tensor has
        one entry per delta."""
        root = tmp_path / "ds"
        _make_dataset(root, ready_eps=[0], total_episodes=1, fps=30,
                      n_frames_per_ep=20)
        src = LeRobotShardSource(root)

        # Deltas of -1/30s, 0, +1/30s → frame offsets -1, 0, 1.
        deltas = [-1.0 / 30.0, 0.0, 1.0 / 30.0]
        reader = SampleReader(
            src, delta_timestamps={"observation.image": deltas},
        )
        item = reader.read(raw_ep=0, frame_in_ep=5)
        assert item["observation.image"].shape[0] == 3

    def test_read_padding_mask_flags_out_of_bounds(self, tmp_path):
        """At frame 0, the -1 delta is out-of-bounds; at frame N-1 the +1
        delta is OOB.  The looked-up value is still clamped to the
        nearest in-bounds frame (no IndexError) — the flag is the only
        signal."""
        root = tmp_path / "ds"
        _make_dataset(root, ready_eps=[0], total_episodes=1, fps=30,
                      n_frames_per_ep=20)
        src = LeRobotShardSource(root)

        deltas = [-1.0 / 30.0, 0.0, 1.0 / 30.0]
        reader = SampleReader(
            src, delta_timestamps={"observation.image": deltas},
        )

        item0 = reader.read(raw_ep=0, frame_in_ep=0)
        mask0 = item0["observation.image_is_pad"]
        assert mask0.tolist() == [True, False, False]

        item_last = reader.read(raw_ep=0, frame_in_ep=19)
        mask_last = item_last["observation.image_is_pad"]
        assert mask_last.tolist() == [False, False, True]

    def test_read_task_string_from_shard(self, tmp_path):
        """Task string comes from ``meta/tasks.jsonl`` via the shard."""
        root = tmp_path / "ds"
        _make_dataset(root, ready_eps=[0], total_episodes=1)
        src = LeRobotShardSource(root)

        reader = SampleReader(src)
        item = reader.read(raw_ep=0, frame_in_ep=0)
        # The ``_make_dataset`` fixture writes ``"task": "synthetic_task"``
        # at ``task_index=0``.
        assert item["task"] == "synthetic_task"


# ---------------------------------------------------------------------------
# Supplied-frames (train path) vs on-demand decode (val path)
# ---------------------------------------------------------------------------


class TestSuppliedVsOnDemandFrames:

    def test_supplied_frames_bypass_decode_cache(self, tmp_path):
        """When ``frames_uint8`` is provided, the reader's internal
        decode LRU stays empty — no unnecessary I/O."""
        root = tmp_path / "ds"
        _make_dataset(root, ready_eps=[0], total_episodes=1, fps=30,
                      n_frames_per_ep=10)
        src = LeRobotShardSource(root)

        reader = SampleReader(src)
        fake_frames = torch.randint(
            0, 256, (10, 3, 32, 32), dtype=torch.uint8,
        )
        reader.read(raw_ep=0, frame_in_ep=3, frames_uint8=fake_frames)
        # Decode cache never populated.
        assert (
            reader._decode_cache is None
            or len(reader._decode_cache) == 0
        )

    def test_on_demand_decode_populates_cache(self, tmp_path):
        """Omitting ``frames_uint8`` triggers a decode-and-cache."""
        root = tmp_path / "ds"
        _make_dataset(root, ready_eps=[0], total_episodes=1)
        src = LeRobotShardSource(root)

        reader = SampleReader(src)
        reader.read(raw_ep=0, frame_in_ep=0)
        assert reader._decode_cache is not None
        assert 0 in reader._decode_cache

    def test_decode_cache_evicts_lru(self, tmp_path):
        """Reader is built with maxsize=2; a third episode evicts the
        least-recently-used one."""
        root = tmp_path / "ds"
        _make_dataset(
            root, ready_eps=[0, 1, 2], total_episodes=3,
        )
        src = LeRobotShardSource(root)

        reader = SampleReader(src, decode_cache_maxsize=2)
        reader.read(raw_ep=0, frame_in_ep=0)
        reader.read(raw_ep=1, frame_in_ep=0)
        reader.read(raw_ep=2, frame_in_ep=0)
        assert set(reader._decode_cache.keys()) == {1, 2}

    def test_supplied_vs_on_demand_produce_identical_samples(self, tmp_path):
        """Cross-path equivalence: the sample at ``(ep, frame)`` must
        match whether frames are supplied pre-decoded or decoded on
        demand.  This is the invariant that lets train/val share the
        reader."""
        root = tmp_path / "ds"
        _make_dataset(
            root, ready_eps=[0], total_episodes=1, fps=30,
            n_frames_per_ep=15,
        )
        src = LeRobotShardSource(root)
        deltas = [-1.0 / 30.0, 0.0, 1.0 / 30.0]

        # Build the pre-decoded frames via the same decode helper the
        # reader uses internally.
        from dataporter.sample_reader import decode_episode_frames_uint8
        video_path = src.episode_video_path(0, src.video_keys[0])
        frames = decode_episode_frames_uint8(
            video_path, 15, int(src.fps),
        )

        r_supplied = SampleReader(
            src, delta_timestamps={"observation.image": deltas},
        )
        r_ondemand = SampleReader(
            src, delta_timestamps={"observation.image": deltas},
        )
        a = r_supplied.read(0, 5, frames_uint8=frames)
        b = r_ondemand.read(0, 5)

        # Video tensors byte-identical.
        assert torch.equal(
            (a["observation.image"] * 255).to(torch.uint8),
            (b["observation.image"] * 255).to(torch.uint8),
        )
        # Padding flag identical.
        assert torch.equal(
            a["observation.image_is_pad"], b["observation.image_is_pad"],
        )
        # Non-video row data identical.
        assert int(a["frame_index"]) == int(b["frame_index"])
        assert a["task"] == b["task"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_read_clamps_against_decoded_frame_count(self, tmp_path):
        """When the caller supplies fewer frames than ``episodes.jsonl``
        declares (possible at the mp4/parquet margin), the video indices
        are clamped to the shorter length — no IndexError."""
        root = tmp_path / "ds"
        _make_dataset(
            root, ready_eps=[0], total_episodes=1, n_frames_per_ep=20,
        )
        src = LeRobotShardSource(root)

        reader = SampleReader(src)
        short_frames = torch.zeros((15, 3, 32, 32), dtype=torch.uint8)
        item = reader.read(raw_ep=0, frame_in_ep=18, frames_uint8=short_frames)
        # Clamped to the shorter length; no IndexError.
        assert item["observation.image"].shape == (1, 3, 32, 32)

    def test_read_handles_cross_episode_cache_reuse(self, tmp_path):
        """Two calls at the same episode reuse the decoded frames from
        the LRU — no double-decode."""
        root = tmp_path / "ds"
        _make_dataset(root, ready_eps=[0], total_episodes=1)
        src = LeRobotShardSource(root)

        reader = SampleReader(src, decode_cache_maxsize=2)
        reader.read(raw_ep=0, frame_in_ep=0)
        cached_tensor = reader._decode_cache[0]
        # Second call: same tensor object (ID check — no re-decode).
        reader.read(raw_ep=0, frame_in_ep=5)
        assert reader._decode_cache[0] is cached_tensor


# ---------------------------------------------------------------------------
# uint8 wire (return_uint8): raw frames out, /255 deferred to the GPU
# ---------------------------------------------------------------------------


class TestReturnUint8:
    """``return_uint8=True`` emits the raw uint8 ``[0, 255]`` window instead
    of float32 ``[0, 1]`` — the dataset half of the uint8-wire optimization.
    The GPU-side DtypeCoordinator (normalize rule) does the ``/255`` upcast,
    so the model still sees identical floats at 1/4 the wire bytes.
    """

    def test_return_uint8_emits_raw_frames_bit_exact_to_float_path(self, tmp_path):
        root = tmp_path / "ds"
        _make_dataset(root, ready_eps=[0], total_episodes=1, fps=30,
                      n_frames_per_ep=15)
        src = LeRobotShardSource(root)
        deltas = [-1.0 / 30.0, 0.0, 1.0 / 30.0]
        frames = torch.randint(0, 256, (15, 3, 32, 32), dtype=torch.uint8)

        r_uint8 = SampleReader(
            src, delta_timestamps={"observation.image": deltas},
            return_uint8=True,
        )
        r_float = SampleReader(
            src, delta_timestamps={"observation.image": deltas},
            return_uint8=False,
        )
        u = r_uint8.read(0, 5, frames_uint8=frames)
        f = r_float.read(0, 5, frames_uint8=frames)

        # uint8 path keeps the raw dtype...
        assert u["observation.image"].dtype == torch.uint8
        # ...and the default float path is the historical float32 [0, 1].
        assert f["observation.image"].dtype == torch.float32
        # The coordinator's GPU-side /255 reproduces the float path EXACTLY.
        assert torch.equal(
            u["observation.image"].to(torch.float32) / 255.0,
            f["observation.image"],
        )
        # Non-video fields are unaffected by the wire format.
        assert torch.equal(
            u["observation.image_is_pad"], f["observation.image_is_pad"],
        )

    def test_return_uint8_single_frame_path(self, tmp_path):
        """The no-delta single-frame branch also honors return_uint8."""
        root = tmp_path / "ds"
        _make_dataset(root, ready_eps=[0], total_episodes=1, n_frames_per_ep=10)
        src = LeRobotShardSource(root)
        frames = torch.randint(0, 256, (10, 3, 32, 32), dtype=torch.uint8)

        reader = SampleReader(src, return_uint8=True)
        item = reader.read(raw_ep=0, frame_in_ep=3, frames_uint8=frames)
        img = item["observation.image"]
        assert img.dtype == torch.uint8
        assert img.shape == (1, 3, 32, 32)
        # Single-frame path selects frame_in_ep exactly.
        assert torch.equal(img[0], frames[3])
