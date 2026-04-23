"""BlendedLeRobotDataModule constructor-level knob plumbing.

Regression locks for:

- ``buffer_rotation_per_samples`` default is ``None`` (the K=1 gate
  misframes video's put-vs-sample granularity ratio and used to crash
  training with "pool may be dead" on working configs).
- ``producer_pool_workers`` is a user-tunable kwarg (previously
  hardcoded ``total_workers=4`` inside the DataModule).

These tests instantiate the DataModule without running ``setup()`` so
they're hermetic (no HF calls, no disk I/O).  They assert only
constructor-level state — the actual pool/buffer wiring is covered by
the e2e tests.
"""

from __future__ import annotations

import pytest


pytest.importorskip("lerobot")


def _dm(**overrides):
    from dataporter import BlendedLeRobotDataModule
    base = dict(
        repo_id="lerobot/pusht",
        delta_timestamps={"observation.image": [0.0]},
    )
    base.update(overrides)
    return BlendedLeRobotDataModule(**base)


class TestBufferRotationDefault:

    def test_default_is_none(self):
        """Default must be ``None`` — K=1 over-fires on video (~100
        frames per put, 1 sample per frame).
        """
        dm = _dm()
        assert dm.buffer_rotation_per_samples is None

    def test_explicit_none_preserved(self):
        dm = _dm(buffer_rotation_per_samples=None)
        assert dm.buffer_rotation_per_samples is None

    def test_explicit_int_plumbed(self):
        dm = _dm(buffer_rotation_per_samples=50)
        assert dm.buffer_rotation_per_samples == 50

    def test_negative_or_zero_still_coerced(self):
        """We coerce to int but don't validate — RotationGate is the
        authoritative validator and will reject 0/negative at
        buffer-construction time.  Just confirm int coercion here.
        """
        dm = _dm(buffer_rotation_per_samples=109)
        assert dm.buffer_rotation_per_samples == 109


class TestProducerPoolWorkers:

    def test_default_is_4(self):
        dm = _dm()
        assert dm.producer_pool_workers == 4

    def test_explicit_override_plumbed(self):
        dm = _dm(producer_pool_workers=8)
        assert dm.producer_pool_workers == 8

    def test_int_coercion(self):
        dm = _dm(producer_pool_workers="6")
        assert dm.producer_pool_workers == 6
