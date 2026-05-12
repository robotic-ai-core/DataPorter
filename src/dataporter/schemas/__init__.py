"""Batch / sample schema contracts for DataPorter.

A :class:`Schema` declares per-field invariants (:class:`FieldSpec`)
and optional per-source extra checks so consumers can validate data
at well-defined pipeline boundaries and fail loud on drift.

Concrete schemas:
  - :class:`TextSampleSpec` — per-sample text contract
    (``input_ids``/``labels``/``loss_mask``/``source_tag``).
  - :class:`VideoActionBatchSpec` — per-batch LeRobot contract
    (``action``/image keys/optional ``*_is_pad`` flags).
"""

from .base import FieldSpec, Schema, SchemaError
from .text import TextSampleSpec
from .video_action import VideoActionBatchSpec
from ._adapters import (
    AddCausalLabels,
    DropExtras,
    EnsureLossMask,
    StampSourceTag,
    ValidateSpec,
    chat_query_adapter,
    pretrain_pad_adapter,
    val_full_adapter,
)

__all__ = [
    "FieldSpec",
    "Schema",
    "SchemaError",
    "TextSampleSpec",
    "VideoActionBatchSpec",
    "AddCausalLabels",
    "DropExtras",
    "EnsureLossMask",
    "StampSourceTag",
    "ValidateSpec",
    "chat_query_adapter",
    "pretrain_pad_adapter",
    "val_full_adapter",
]
