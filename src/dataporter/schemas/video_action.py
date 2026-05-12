"""Video-action batch schema for LeRobot-style training.

``VideoActionBatchSpec`` declares the post-collate batch contract for
``BlendedLeRobotDataModule``: action vectors, image-key tensors,
optional ``*_is_pad`` flags, and ``source_tag``.

Batches contain a leading batch dim ``B`` that is left as a wildcard —
``FieldSpec.shape`` entries use ``None`` for any wildcard dim.

Per-project subclasses can pin the image-key dtype/shape to match
their specific pipeline (e.g. uint8 [B,T,C,H,W] for ProtoWorld /
ternary_wm).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Mapping

import torch

from .base import FieldSpec, Schema, SchemaError


@dataclass(frozen=True)
class VideoActionBatchSpec(Schema):
    """Per-batch video-action contract.

    Args:
        time_steps: ``T`` — number of frames per sample
            (``delta_timestamps`` window length).
        action_dim: ``A`` — action vector dim.
        image_keys: tuple of image-feature names emitted by the
            datamodule (e.g. ``("observation.image",)`` for pusht).
            Each gets a FieldSpec checking dtype + ``(B, T, C, H, W)``
            shape via ``image_channels``/``image_height``/``image_width``.
        image_channels: ``C`` (typically 3 for RGB).
        image_height: ``H``.  ``None`` accepts any height.
        image_width: ``W``.  ``None`` accepts any width.
        image_dtype: dtype emitted by ``SampleReader`` for image keys.
            Defaults to ``torch.uint8`` — the wire dtype.  Working dtype
            (float32, bf16) is applied by the dtype coordinator *after*
            batch transfer.
        require_pad_flags: when ``True``, every image key + ``action``
            field must have a matching ``<key>_is_pad`` boolean tensor.
    """

    time_steps: int
    action_dim: int
    image_keys: tuple[str, ...] = ("observation.image",)
    image_channels: int = 3
    image_height: int | None = None
    image_width: int | None = None
    image_dtype: torch.dtype = torch.uint8
    require_pad_flags: bool = False

    REQUIRED_KEYS: ClassVar[tuple[str, ...]] = ("action",)
    # KNOWN_SOURCES intentionally empty — source_tag is informational
    # (stamped per-sample by SourceTagDataset) and not a gate-check.

    # FIELDS is built per-instance in __post_init__ so per-instance
    # image_keys flow into per-key FieldSpec entries. The ClassVar here
    # is a no-op placeholder; instance attribute shadows it.
    FIELDS: ClassVar[dict[str, FieldSpec]] = {}
    _INVARIANTS: ClassVar[dict[str, Callable]] = {}

    def __post_init__(self) -> None:
        fields: dict[str, FieldSpec] = {
            "action": FieldSpec(
                dtype=torch.float32,
                shape=(None, "time_steps", "action_dim"),
            ),
        }
        required: list[str] = ["action", *self.image_keys]
        for key in self.image_keys:
            fields[key] = FieldSpec(
                dtype=self.image_dtype,
                shape=(
                    None,
                    "time_steps",
                    "image_channels",
                    "image_height",
                    "image_width",
                ),
            )
        if self.require_pad_flags:
            pad_shape = (None, "time_steps")
            fields["action_is_pad"] = FieldSpec(
                dtype=torch.bool, shape=pad_shape,
            )
            required.append("action_is_pad")
            for key in self.image_keys:
                fields[f"{key}_is_pad"] = FieldSpec(
                    dtype=torch.bool, shape=pad_shape,
                )
                required.append(f"{key}_is_pad")
        # Frozen-dataclass: route around the immutability guard to
        # install FIELDS + REQUIRED_KEYS as instance attributes that
        # shadow the ClassVar declarations above.
        object.__setattr__(self, "FIELDS", fields)
        object.__setattr__(self, "REQUIRED_KEYS", tuple(required))
