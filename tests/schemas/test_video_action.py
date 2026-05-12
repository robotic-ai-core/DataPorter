"""Unit tests for VideoActionBatchSpec — post-collate LeRobot batches."""

from __future__ import annotations

import pytest
import torch

from dataporter.schemas import SchemaError, VideoActionBatchSpec


def _mk_batch(
    B: int = 4,
    T: int = 8,
    A: int = 2,
    C: int = 3,
    H: int = 64,
    W: int = 64,
    image_key: str = "observation.image",
    image_dtype: torch.dtype = torch.uint8,
    with_pad_flags: bool = False,
) -> dict:
    batch = {
        "action": torch.zeros(B, T, A, dtype=torch.float32),
        image_key: torch.zeros(B, T, C, H, W, dtype=image_dtype),
    }
    if with_pad_flags:
        batch["action_is_pad"] = torch.zeros(B, T, dtype=torch.bool)
        batch[f"{image_key}_is_pad"] = torch.zeros(B, T, dtype=torch.bool)
    return batch


# ----- happy path -----------------------------------------------------------


def test_validate_compliant_batch():
    spec = VideoActionBatchSpec(time_steps=8, action_dim=2)
    spec.validate(_mk_batch())


def test_validate_compliant_batch_with_custom_image_dims():
    spec = VideoActionBatchSpec(
        time_steps=4, action_dim=7,
        image_channels=1, image_height=32, image_width=32,
    )
    spec.validate(_mk_batch(T=4, A=7, C=1, H=32, W=32))


def test_validate_compliant_batch_with_pad_flags():
    spec = VideoActionBatchSpec(
        time_steps=8, action_dim=2, require_pad_flags=True,
    )
    spec.validate(_mk_batch(with_pad_flags=True))


def test_validate_multiple_image_keys():
    spec = VideoActionBatchSpec(
        time_steps=4, action_dim=2,
        image_keys=("observation.image.top", "observation.image.side"),
    )
    batch = {
        "action": torch.zeros(2, 4, 2, dtype=torch.float32),
        "observation.image.top": torch.zeros(2, 4, 3, 32, 32, dtype=torch.uint8),
        "observation.image.side": torch.zeros(2, 4, 3, 32, 32, dtype=torch.uint8),
    }
    spec.validate(batch)


def test_height_width_wildcard_passes_any():
    spec = VideoActionBatchSpec(
        time_steps=4, action_dim=2,
        image_height=None, image_width=None,
    )
    spec.validate(_mk_batch(T=4, A=2, H=240, W=320))


# ----- violations -----------------------------------------------------------


def test_missing_action_raises():
    spec = VideoActionBatchSpec(time_steps=8, action_dim=2)
    batch = _mk_batch()
    del batch["action"]
    with pytest.raises(SchemaError, match="missing required keys.*action"):
        spec.validate(batch)


def test_missing_image_key_raises():
    spec = VideoActionBatchSpec(time_steps=8, action_dim=2)
    batch = _mk_batch()
    del batch["observation.image"]
    with pytest.raises(SchemaError, match="missing required keys.*observation"):
        spec.validate(batch)


def test_wrong_action_dtype_raises():
    spec = VideoActionBatchSpec(time_steps=8, action_dim=2)
    batch = _mk_batch()
    batch["action"] = batch["action"].to(torch.float64)
    with pytest.raises(SchemaError, match="action dtype"):
        spec.validate(batch)


def test_wrong_image_dtype_raises():
    spec = VideoActionBatchSpec(time_steps=8, action_dim=2)
    batch = _mk_batch()
    batch["observation.image"] = batch["observation.image"].to(torch.float32)
    with pytest.raises(SchemaError, match="dtype"):
        spec.validate(batch)


def test_wrong_time_steps_raises():
    spec = VideoActionBatchSpec(time_steps=8, action_dim=2)
    batch = _mk_batch(T=10)
    with pytest.raises(SchemaError, match="dim 1"):
        spec.validate(batch)


def test_wrong_action_dim_raises():
    spec = VideoActionBatchSpec(time_steps=8, action_dim=2)
    batch = _mk_batch(A=5)
    with pytest.raises(SchemaError, match="action.*dim 2"):
        spec.validate(batch)


def test_pin_image_dtype_to_working_dtype():
    """Project that has already cast images to bf16 via wire→working
    coordinator can validate with that working dtype."""
    spec = VideoActionBatchSpec(
        time_steps=4, action_dim=2, image_dtype=torch.bfloat16,
    )
    batch = _mk_batch(T=4, A=2, image_dtype=torch.bfloat16)
    spec.validate(batch)
    # uint8 wire image now fails — the spec is pinned to bf16.
    with pytest.raises(SchemaError, match="dtype"):
        spec.validate(_mk_batch(T=4, A=2, image_dtype=torch.uint8))


def test_pad_flags_required_but_absent_raises():
    spec = VideoActionBatchSpec(
        time_steps=8, action_dim=2, require_pad_flags=True,
    )
    batch = _mk_batch(with_pad_flags=False)
    with pytest.raises(SchemaError, match="missing required keys"):
        spec.validate(batch)


def test_pad_flags_wrong_dtype_raises():
    spec = VideoActionBatchSpec(
        time_steps=8, action_dim=2, require_pad_flags=True,
    )
    batch = _mk_batch(with_pad_flags=True)
    batch["action_is_pad"] = batch["action_is_pad"].to(torch.uint8)
    with pytest.raises(SchemaError, match="dtype"):
        spec.validate(batch)


# ----- batch-size wildcard --------------------------------------------------


def test_batch_size_is_wildcard():
    spec = VideoActionBatchSpec(time_steps=8, action_dim=2)
    for B in (1, 4, 16, 128):
        spec.validate(_mk_batch(B=B))
