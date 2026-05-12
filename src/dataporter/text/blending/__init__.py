"""Text blending primitives: multi-source weighted draw + mix schedules.

These are policy classes — the orchestration (the surrounding
``LightningDataModule``) lives in the project that consumes them.

Public surface:
  - :class:`WeightedMultiSourceDataset` — stochastic N-way draw with
    shared-memory mutable weights.
  - :class:`BlendedTextDataset` — two-way (pretrain + chat) mix with
    shared-memory mutable ratio.
  - :class:`PretrainBlendScheduleCallback` — drives
    ``WeightedMultiSourceDataset`` weights along piecewise-linear curves.
  - :class:`MixingScheduleCallback` — anneals
    ``BlendedTextDataset.chat_ratio`` linearly over a step range.
"""

from .blended_dataset import BlendedTextDataset
from .callbacks import MixingScheduleCallback, PretrainBlendScheduleCallback
from .weighted_multi_source import WeightedMultiSourceDataset

__all__ = [
    "BlendedTextDataset",
    "MixingScheduleCallback",
    "PretrainBlendScheduleCallback",
    "WeightedMultiSourceDataset",
]
