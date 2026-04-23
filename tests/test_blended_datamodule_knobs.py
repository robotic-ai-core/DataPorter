"""BlendedLeRobotDataModule constructor-level knob plumbing.

``producer_pool_workers`` is user-tunable; the hardcoded
``total_workers=4`` inside the DataModule is gone.  The buffer's
rotation gate is now flow-balance driven (frame-count based) with
no tuning knob — see :class:`RotationGate`.
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
