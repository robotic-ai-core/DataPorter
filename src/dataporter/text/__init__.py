"""Random-access and streaming text data sources.

Two access patterns coexist:

- **Random-access** (this module): ``ParquetTokenDataset`` (preload or
  row-group LRU streaming), ``ChatDataset`` (preloaded chat examples)
  — best for datasets that fit in RAM or have stable file layouts.

- **Producer-pool streaming** (top-level :mod:`dataporter`):
  ``TokenShuffleBuffer`` / ``TextProducerPool`` / ``RawTextSource`` —
  best for large pretokenized pretrain corpora that need continuous
  prefetch + shuffle.

Pick by workload, not by source kind.
"""

from .chunking import TokenChunker
from .parquet_dataset import ParquetTokenDataset
from .stream_dataset import PretrainStreamDataset
from .types import TextSample
from .chat import ChatDataset, ChatStreamDataset, apply_chat_template
from .blending import (
    BlendedTextDataset,
    MixingScheduleCallback,
    PretrainBlendScheduleCallback,
    ScheduledBlendDataset,
    SourceScheduleCallback,
    WeightedMultiSourceDataset,
)

__all__ = [
    "TokenChunker",
    "ParquetTokenDataset",
    "PretrainStreamDataset",
    "TextSample",
    "ChatDataset",
    "ChatStreamDataset",
    "apply_chat_template",
    "ScheduledBlendDataset",
    "SourceScheduleCallback",
    "BlendedTextDataset",
    "MixingScheduleCallback",
    "PretrainBlendScheduleCallback",
    "WeightedMultiSourceDataset",
]
