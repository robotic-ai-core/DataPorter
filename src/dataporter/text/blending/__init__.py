"""Text blending primitives: weighted draw + mix schedules.

The canonical primitive is :class:`ScheduledBlendDataset` paired with
:class:`SourceScheduleCallback`. The other four classes are
backward-compatibility wrappers retained through Phase 4; they emit
``DeprecationWarning`` on construction and will be removed in Phase 5.

Public surface:
  - :class:`ScheduledBlendDataset` — N-way stochastic draw over named
    sources with shared-memory mutable weights.
  - :class:`SourceScheduleCallback` — drives ``ScheduledBlendDataset``
    weights along piecewise-linear curves; schedules keyed by
    ``source_name`` (preferred) or ``source_idx``.

Backward-compat wrappers (deprecated):
  - :class:`WeightedMultiSourceDataset` — old N-way without names.
  - :class:`BlendedTextDataset` — old 2-way pretrain/chat.
  - :class:`PretrainBlendScheduleCallback` — old N-way schedule.
  - :class:`MixingScheduleCallback` — old 2-way chat-ratio anneal.
"""

from .blended_dataset import BlendedTextDataset
from .callbacks import MixingScheduleCallback, PretrainBlendScheduleCallback
from .scheduled_blend import ScheduledBlendDataset
from .source_schedule import SourceScheduleCallback
from .weighted_multi_source import WeightedMultiSourceDataset

__all__ = [
    "ScheduledBlendDataset",
    "SourceScheduleCallback",
    "BlendedTextDataset",
    "MixingScheduleCallback",
    "PretrainBlendScheduleCallback",
    "WeightedMultiSourceDataset",
]
