"""Type hints for text samples.

``TextSample`` is a ``total=False`` TypedDict — raw producers (e.g.
``ParquetTokenDataset``) emit only ``input_ids``/``labels``; the adapter
chain (``dataporter.schemas._adapters``) normalises to the full
``{input_ids, labels, loss_mask, source_tag}`` contract before samples
reach a datamodule boundary.

Use :class:`dataporter.schemas.TextSampleSpec` for runtime contract
enforcement.
"""

from __future__ import annotations

from typing import TypedDict

import torch


class TextSample(TypedDict, total=False):
    """Data contract for text/language-modeling samples."""

    input_ids: torch.Tensor  # [seq_len], long
    labels: torch.Tensor  # [seq_len], long — required post-adapter
    loss_mask: torch.Tensor  # [seq_len], bool — required post-adapter
    source_tag: str  # one of TextSampleSpec.KNOWN_SOURCES — required post-adapter
