"""Tests exposing multi-source collision bugs in the shuffle buffer pipeline.

Three bugs are demonstrated:

1. **Episode index collision in _ep_to_source** (lerobot_shuffle_buffer_dataset.py):
   When two sources both have episodes starting at 0, source B's entries
   overwrite source A's in the flat dict. Episode 0 always maps to the
   last source, so source A's episodes are entirely unreachable.

2. **Buffer key collision -- no source disambiguation** (shuffle_buffer.py):
   The key is the raw episode index. The buffer CAN hold two items with
   key=0 in separate slots (ring buffer), but ``keys()`` returns duplicate
   values and there is no way to tell which source produced a given sample.
   Combined with bug 1, source A's frames are always attributed to source B.

3. **Buffer composition drift in ProducerPool** (producer_pool.py):
   Thread allocation respects weights, but actual buffer composition
   depends on decode speed. Sources with shorter episodes fill more slots
   than their weight warrants.

Tests marked ``xfail`` are expected to FAIL with the current code, serving
as a regression gate. Once the collision bugs are fixed, remove the xfail
markers and the tests should pass.
"""

from __future__ import annotations

import random
import time
from collections import Counter

import pytest
import torch

from dataporter.producer_pool import AsyncProducer, ProducerPool
from dataporter.shuffle_buffer import ShuffleBuffer
from dataporter.lerobot_shuffle_buffer_dataset import LeRobotShuffleBufferDataset
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_ep_to_source(sources: list[dict]) -> dict[int, int]:
    """Reproduce the _ep_to_source construction from
    LeRobotShuffleBufferDataset.__init__ (lines 82-89).
    """
    ep_to_source: dict[int, int] = {}
    for src_idx, src in enumerate(sources):
        for ep_idx in src["train_episode_indices"]:
            ep_to_source[ep_idx] = src_idx
    return ep_to_source


def _make_frames(
    n_frames: int,
    fill_value: int,
    c: int = 3,
    h: int = 4,
    w: int = 4,
) -> torch.Tensor:
    """Create uint8 frames filled with a constant value for easy identification."""
    return torch.full((n_frames, c, h, w), fill_value, dtype=torch.uint8)


def _make_mock_dataset(
    num_episodes: int = 5,
    frames_per_episode: int = 10,
    fps: int = 10,
) -> MagicMock:
    """Lightweight mock FastLeRobotDataset for source attribution tests."""
    ds = MagicMock()
    ds.fps = fps
    ds.tolerance_s = 0.04
    ds.video_backend = "pyav"
    ds.root = MagicMock()
    ds.meta = MagicMock()
    ds.meta.fps = fps
    ds.meta.video_keys = ["observation.image"]
    ds.meta.tasks = {0: "task_0"}

    froms = [i * frames_per_episode for i in range(num_episodes)]
    tos = [(i + 1) * frames_per_episode for i in range(num_episodes)]
    ds.episode_data_index = {
        "from": torch.tensor(froms),
        "to": torch.tensor(tos),
    }

    def hf_getitem(idx):
        ep_idx = idx // frames_per_episode
        frame_idx = idx % frames_per_episode
        return {
            "episode_index": torch.tensor(ep_idx),
            "frame_index": torch.tensor(frame_idx),
            "timestamp": torch.tensor(frame_idx / fps),
            "task_index": torch.tensor(0),
            "index": torch.tensor(idx),
            "action": torch.randn(2),
            "observation.state": torch.randn(4),
        }

    class MockHfDataset:
        def __getitem__(self, idx):
            return hf_getitem(idx)

    ds.hf_dataset = MockHfDataset()

    def query_hf_dataset(query_indices):
        result = {}
        for key, indices in query_indices.items():
            if key == "observation.image":
                continue
            rows = [hf_getitem(i) for i in indices]
            result[key] = torch.stack([r[key] for r in rows])
        return result

    ds._query_hf_dataset = query_hf_dataset

    return ds


# ---------------------------------------------------------------------------
# Test 1: _ep_to_source collision
# ---------------------------------------------------------------------------

class TestEpToSourceCollision:

    @pytest.mark.xfail(
        reason="Bug: episode index collision -- source B overwrites source A "
               "in flat _ep_to_source dict (no source namespacing)",
        strict=True,
    )
    def test_ep_to_source_collision_detected(self):
        """Two sources with overlapping episode indices [0..4] should each
        have their own entries in _ep_to_source. With the current flat dict,
        source B's entries silently overwrite source A's.

        Expected: 10 entries (5 per source, namespaced by source).
        Actual:    5 entries (source B overwrites source A).
        """
        sources = [
            {"train_episode_indices": [0, 1, 2, 3, 4]},  # Source A
            {"train_episode_indices": [0, 1, 2, 3, 4]},  # Source B
        ]

        ep_to_source = _build_ep_to_source(sources)

        # After the bug, len == 5 (all mapped to source 1).
        # Correct behavior requires 10 entries or a composite key.
        assert len(ep_to_source) == 10, (
            f"Expected 10 entries (5 per source), got {len(ep_to_source)}. "
            f"Source B overwrote source A for overlapping episode indices."
        )

    @pytest.mark.xfail(
        reason="Bug: episode index collision -- all overlapping episodes "
               "map to the last source",
        strict=True,
    )
    def test_all_sources_reachable_via_lookup(self):
        """Every source should be reachable via _ep_to_source lookup.
        With the collision bug, source 0 is entirely unreachable because
        source 1's episodes overwrite its entries.
        """
        sources = [
            {"train_episode_indices": [0, 1, 2, 3, 4]},  # Source A (idx=0)
            {"train_episode_indices": [0, 1, 2, 3, 4]},  # Source B (idx=1)
        ]

        ep_to_source = _build_ep_to_source(sources)

        reachable_sources = set(ep_to_source.values())
        assert 0 in reachable_sources, (
            f"Source 0 is unreachable! All episodes map to sources: "
            f"{reachable_sources}. Source B overwrote source A."
        )
        assert 1 in reachable_sources


# ---------------------------------------------------------------------------
# Test 2: Buffer key ambiguity (no source disambiguation)
# ---------------------------------------------------------------------------

class TestBufferKeyCollision:

    @pytest.mark.xfail(
        reason="Bug: buffer keys() returns duplicate episode indices with no "
               "source tag -- impossible to distinguish source A's ep 0 from "
               "source B's ep 0",
        strict=True,
    )
    def test_buffer_keys_unique_per_source(self):
        """When two sources write episodes with the same indices, the buffer's
        keys() should return source-disambiguated identifiers so the consumer
        can tell which source produced each item.

        With the current code, keys() returns [0, 0] -- two indistinguishable
        entries. A correct implementation would return distinct keys like
        (0, 0) and (1, 0), or namespaced integers like 0 and 1000000.
        """
        buf = ShuffleBuffer(
            capacity=10, max_frames=5, channels=3, height=4, width=4,
        )

        buf.put(0, _make_frames(5, fill_value=0))    # Source A, episode 0
        buf.put(0, _make_frames(5, fill_value=255))   # Source B, episode 0

        keys = buf.keys()
        # The buffer holds 2 items, both with key=0.
        assert len(keys) == 2, f"Expected 2 items, got {len(keys)}"

        # The keys must be UNIQUE so _ep_to_source can disambiguate.
        # Currently both are 0 -- this is the bug.
        assert len(set(keys)) == 2, (
            f"Buffer has 2 items but only {len(set(keys))} unique key(s): "
            f"{keys}. Source A's ep 0 and source B's ep 0 are "
            f"indistinguishable."
        )

    @pytest.mark.xfail(
        reason="Bug: sample() returns raw episode index with no source tag -- "
               "_ep_to_source always maps to the last source that registered "
               "that episode index",
        strict=True,
    )
    def test_sampled_key_identifies_source(self):
        """When sampling from a buffer with overlapping keys, the returned
        key must uniquely identify the source so _ep_to_source can route
        correctly.

        Source A writes ep 0 with pixel=50, source B writes ep 0 with pixel=200.
        When we sample ep 0 with pixel=50, _ep_to_source should map it to
        source A (idx=0). But with the collision, it always maps to source B.
        """
        buf = ShuffleBuffer(
            capacity=10, max_frames=5, channels=3, height=4, width=4,
        )

        buf.put(0, _make_frames(5, fill_value=50))    # Source A
        buf.put(0, _make_frames(5, fill_value=200))   # Source B

        # Build _ep_to_source the way the dataset does
        sources = [
            {"train_episode_indices": [0]},  # Source A
            {"train_episode_indices": [0]},  # Source B
        ]
        ep_to_source = _build_ep_to_source(sources)

        # Sample many times. When we get source A's frames (pixel=50),
        # ep_to_source should map to source 0. When we get source B's
        # frames (pixel=200), it should map to source 1.
        rng = random.Random(42)
        source_a_attributed_correctly = False
        for _ in range(200):
            key, frames = buf.sample(rng)
            val = frames[0, 0, 0, 0].item()
            mapped_source = ep_to_source.get(key)

            if val == 50 and mapped_source == 0:
                source_a_attributed_correctly = True

        assert source_a_attributed_correctly, (
            "Source A's frames (pixel=50) were never correctly attributed "
            "to source 0. The collision in _ep_to_source always maps key=0 "
            "to source 1 (the last writer)."
        )


# ---------------------------------------------------------------------------
# Test 3: Source attribution accuracy (end-to-end dataset test)
# ---------------------------------------------------------------------------

class TestSourceAttributionAccuracy:

    @pytest.mark.xfail(
        reason="Bug: combined ep_to_source + buffer key collision causes "
               "source A to be entirely unreachable from the dataset",
        strict=True,
    )
    def test_source_attribution_accuracy(self):
        """Each sampled item's source (as determined by _ep_to_source) must
        include BOTH sources when both have data in the buffer.

        Source A: episodes 0-4, frames filled with pixel value 50.
        Source B: episodes 0-4, frames filled with pixel value 200.

        After the collision, _ep_to_source maps all episodes to source B
        (idx=1), so source A (idx=0) is never returned.
        """
        buf = ShuffleBuffer(
            capacity=20, max_frames=10, channels=3, height=4, width=4,
        )
        ds_a = _make_mock_dataset(num_episodes=5, frames_per_episode=10)
        ds_b = _make_mock_dataset(num_episodes=5, frames_per_episode=10)

        # Fill buffer: source A episodes 0-4 with pixel=50
        for ep in range(5):
            buf.put(ep, _make_frames(10, fill_value=50))

        # Fill buffer: source B episodes 0-4 with pixel=200
        for ep in range(5):
            buf.put(ep, _make_frames(10, fill_value=200))

        sources = [
            {
                "dataset": ds_a,
                "train_episode_indices": [0, 1, 2, 3, 4],
                "transform": None,
            },
            {
                "dataset": ds_b,
                "train_episode_indices": [0, 1, 2, 3, 4],
                "transform": None,
            },
        ]

        dataset = LeRobotShuffleBufferDataset(
            buffer=buf,
            sources=sources,
            delta_timestamps={"observation.image": [0.0]},
            epoch_length=200,
            image_keys=["observation.image"],
        )

        # Sample many items and track which source indices are reached
        source_indices_sampled = set()
        for _ in range(200):
            item = dataset[0]
            ep_idx = item["episode_index"].item()
            src_idx = dataset._ep_to_source.get(ep_idx)
            if src_idx is not None:
                source_indices_sampled.add(src_idx)

        assert 0 in source_indices_sampled, (
            f"Source A (idx=0) was never sampled! Only sources "
            f"{source_indices_sampled} were reachable. The collision bug "
            f"makes source A's episodes invisible -- all episode keys map "
            f"to source B (idx=1)."
        )
        assert 1 in source_indices_sampled


# ---------------------------------------------------------------------------
# Test 4: Composition ratio under equal decode speed
# ---------------------------------------------------------------------------

class TestCompositionRatio:

    def test_composition_ratio_under_equal_decode_speed(self):
        """Two sources with equal episode lengths but 3:1 weight ratio.
        Fill buffer via ProducerPool, sample 1000 items, verify the
        source ratio is within 25% of target (2.25:1 to 3.75:1).

        Uses non-overlapping episode ranges to avoid the key collision
        bugs. Validates that weighted round-robin dispatch in ProducerPool
        produces the intended blend ratio when decode speeds are identical.
        """
        buf = ShuffleBuffer(
            capacity=60, max_frames=5, channels=3, height=4, width=4,
        )

        def decode_a(ep_idx: int) -> torch.Tensor:
            return _make_frames(5, fill_value=10)

        def decode_b(ep_idx: int) -> torch.Tensor:
            return _make_frames(5, fill_value=20)

        # Non-overlapping episode ranges to avoid key collision
        producer_a = AsyncProducer(
            "A", decode_a, list(range(0, 50)), weight=3.0,
        )
        producer_b = AsyncProducer(
            "B", decode_b, list(range(1000, 1050)), weight=1.0,
        )

        pool = ProducerPool(
            buf,
            producers=[producer_a, producer_b],
            total_workers=2,
            warmup_target=40,
        )
        pool.start()
        pool.wait_for_warmup(timeout=30)
        # Let the buffer churn a bit to reach steady state
        time.sleep(0.5)
        pool.stop()

        # Sample 1000 items from the buffer
        rng = random.Random(42)
        counts = Counter()
        for _ in range(1000):
            key, _ = buf.sample(rng)
            source = "A" if key < 1000 else "B"
            counts[source] += 1

        a_count = counts.get("A", 0)
        b_count = counts.get("B", 0)
        total = a_count + b_count

        assert total == 1000
        assert b_count > 0, "Source B produced zero samples"

        ratio = a_count / b_count
        # 3:1 weight => expect ratio in [2.25, 3.75] (25% tolerance)
        assert 2.25 <= ratio <= 3.75, (
            f"Expected ~3:1 ratio, got {ratio:.2f}:1 "
            f"(A={a_count}, B={b_count})"
        )


# ---------------------------------------------------------------------------
# Test 5: All sources represented in buffer (via ProducerPool)
# ---------------------------------------------------------------------------

class TestAllSourcesRepresented:

    @pytest.mark.xfail(
        reason="Bug: with overlapping episode indices, _ep_to_source maps all "
               "episodes to the last source. Even though both sources' frames "
               "exist in the buffer, the dataset can only route to source B.",
        strict=True,
    )
    def test_all_sources_represented_in_dataset_output(self):
        """Two sources, each with episodes [0..4], fed via ProducerPool.
        After warmup, the LeRobotShuffleBufferDataset should produce
        samples attributed to BOTH sources.

        With the _ep_to_source collision, all samples are attributed to
        source B (the last registered source for episodes 0-4).
        """
        buf = ShuffleBuffer(
            capacity=20, max_frames=5, channels=3, height=4, width=4,
        )

        def decode_a(ep_idx: int) -> torch.Tensor:
            return _make_frames(5, fill_value=50)

        def decode_b(ep_idx: int) -> torch.Tensor:
            return _make_frames(5, fill_value=200)

        # OVERLAPPING episode indices -- triggers the collision bug
        producer_a = AsyncProducer(
            "A", decode_a, list(range(5)), weight=1.0,
        )
        producer_b = AsyncProducer(
            "B", decode_b, list(range(5)), weight=1.0,
        )

        pool = ProducerPool(
            buf,
            producers=[producer_a, producer_b],
            total_workers=2,
            warmup_target=8,
        )
        pool.start()
        pool.wait_for_warmup(timeout=30)
        time.sleep(0.3)
        pool.stop()

        ds_a = _make_mock_dataset(num_episodes=5, frames_per_episode=5)
        ds_b = _make_mock_dataset(num_episodes=5, frames_per_episode=5)

        dataset = LeRobotShuffleBufferDataset(
            buffer=buf,
            sources=[
                {
                    "dataset": ds_a,
                    "train_episode_indices": list(range(5)),
                    "transform": None,
                },
                {
                    "dataset": ds_b,
                    "train_episode_indices": list(range(5)),
                    "transform": None,
                },
            ],
            delta_timestamps={"observation.image": [0.0]},
            epoch_length=200,
            image_keys=["observation.image"],
        )

        # Track which source indices are reached
        source_indices_sampled = set()
        for _ in range(200):
            item = dataset[0]
            ep_idx = item["episode_index"].item()
            src_idx = dataset._ep_to_source.get(ep_idx)
            if src_idx is not None:
                source_indices_sampled.add(src_idx)

        assert source_indices_sampled == {0, 1}, (
            f"Expected both sources {{0, 1}}, but only {source_indices_sampled} "
            f"were reachable. The _ep_to_source collision maps all episodes "
            f"to source 1."
        )


# ---------------------------------------------------------------------------
# Test 6: Namespaced episodes -- the "after fix" test
# ---------------------------------------------------------------------------

class TestNamespacedEpisodes:

    def test_namespaced_episodes_no_collision(self):
        """Validates the intended behavior AFTER the fix: when episode
        indices are namespaced (source 0 uses eps 0-4, source 1 uses
        eps 5-9), there are no collisions in _ep_to_source, buffer keys,
        or sampled data.

        This test should PASS even with the current buggy code, because
        it manually avoids collisions by using non-overlapping episode
        ranges.
        """
        # -- _ep_to_source: no collision with non-overlapping indices --
        sources = [
            {"train_episode_indices": [0, 1, 2, 3, 4]},
            {"train_episode_indices": [5, 6, 7, 8, 9]},
        ]
        ep_to_source = _build_ep_to_source(sources)

        assert len(ep_to_source) == 10, (
            f"Expected 10 entries, got {len(ep_to_source)}"
        )
        for ep in range(5):
            assert ep_to_source[ep] == 0
        for ep in range(5, 10):
            assert ep_to_source[ep] == 1

        # -- Buffer: no collision with non-overlapping keys --
        buf = ShuffleBuffer(
            capacity=20, max_frames=5, channels=3, height=4, width=4,
        )

        source_a_fill = 50
        source_b_fill = 200

        for ep in range(5):
            buf.put(ep, _make_frames(5, fill_value=source_a_fill))
        for ep in range(5, 10):
            buf.put(ep, _make_frames(5, fill_value=source_b_fill))

        assert len(buf) == 10

        # -- Sampled data: correct source attribution --
        rng = random.Random(42)
        for _ in range(100):
            key, frames = buf.sample(rng)
            val = frames[0, 0, 0, 0].item()
            src_idx = ep_to_source[key]
            expected_fill = source_a_fill if src_idx == 0 else source_b_fill
            assert val == expected_fill, (
                f"Episode {key} mapped to source {src_idx}, expected fill "
                f"{expected_fill} but got {val}"
            )

    def test_namespaced_buffer_via_producer_pool(self):
        """ProducerPool with non-overlapping episode ranges: both sources
        coexist in the buffer. Validates that the pipeline works correctly
        when the collision is avoided at the episode-index level.
        """
        buf = ShuffleBuffer(
            capacity=20, max_frames=5, channels=3, height=4, width=4,
        )

        def decode_a(ep_idx: int) -> torch.Tensor:
            return _make_frames(5, fill_value=50)

        def decode_b(ep_idx: int) -> torch.Tensor:
            return _make_frames(5, fill_value=200)

        producer_a = AsyncProducer(
            "A", decode_a, list(range(0, 5)), weight=1.0,
        )
        producer_b = AsyncProducer(
            "B", decode_b, list(range(100, 105)), weight=1.0,
        )

        pool = ProducerPool(
            buf,
            producers=[producer_a, producer_b],
            total_workers=2,
            warmup_target=8,
        )
        pool.start()
        pool.wait_for_warmup(timeout=30)
        time.sleep(0.3)
        pool.stop()

        rng = random.Random(42)
        found_a = False
        found_b = False
        for _ in range(200):
            key, frames = buf.sample(rng)
            val = frames[0, 0, 0, 0].item()
            if val == 50:
                found_a = True
            elif val == 200:
                found_b = True

        assert found_a, "Source A frames (pixel=50) not found in buffer"
        assert found_b, "Source B frames (pixel=200) not found in buffer"
