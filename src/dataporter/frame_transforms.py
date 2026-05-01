"""Picklable frame-level transforms for the ShuffleBuffer pipeline.

The ShuffleBuffer allocates shared memory at a fixed ``[capacity × T × C
× H × W]`` shape at construction time.  At source resolution (e.g.
224x224) with large capacity and long episodes this blows up quickly
(capacity=2000, T=264, 224x224 uint8 = ~74 GB of shm).  Producer-side
transforms let us write frames into the buffer at *training* resolution
instead — same buffer capacity, a fraction of the shm.

Design:

- Each transform is a picklable ``Callable[[Tensor], Tensor]``.  Classes,
  not lambdas (the spawn child needs to unpickle them).
- Transforms expose an optional ``output_shape(input_shape) ->
  output_shape`` method so the DataModule can compute the buffer's
  allocated shape without probing.  Transforms that don't expose it get
  probed with a dummy tensor at setup time via :func:`probe_output_shape`.

Example::

    from dataporter import ResizeFrames, FrameCompose, BlendedLeRobotDataModule

    dm = BlendedLeRobotDataModule(
        repo_id="neiltan/lewm-pusht-224x224-full",
        producer_transform=ResizeFrames(96, 96),
        ...,
    )

    # Chain transforms if needed
    dm = BlendedLeRobotDataModule(
        ...,
        producer_transform=FrameCompose([
            ResizeFrames(120, 120),
            CenterCropFrames(96, 96),  # hypothetical
        ]),
    )
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

import torch

logger = logging.getLogger(__name__)


def _interpolate_bilinear_uint8(
    frames_uint8: torch.Tensor, height: int, width: int,
) -> torch.Tensor:
    """Bilinear downsample ``[T, C, H, W]`` uint8 frames to ``(height, width)``.

    Round-trips through float32 since ``F.interpolate`` doesn't accept
    uint8 directly; clamps back to the valid byte range.
    """
    if frames_uint8.shape[-2] == height and frames_uint8.shape[-1] == width:
        return frames_uint8
    frames_f = frames_uint8.float()
    frames_resized = torch.nn.functional.interpolate(
        frames_f, size=(height, width), mode="bilinear", align_corners=False,
    )
    return frames_resized.clamp_(0, 255).to(torch.uint8)


class ResizeFrames:
    """Bilinear-downsample ``[T, C, H, W]`` uint8 frames.

    The canonical producer-side transform: use to make the ShuffleBuffer
    allocate shm at training resolution instead of source resolution.
    """

    def __init__(self, height: int, width: int):
        self.height = int(height)
        self.width = int(width)

    def __call__(self, frames_uint8: torch.Tensor) -> torch.Tensor:
        return _interpolate_bilinear_uint8(
            frames_uint8, self.height, self.width,
        )

    def output_shape(
        self, input_shape: Sequence[int],
    ) -> tuple[int, ...]:
        """``[..., H, W]`` → ``[..., target_h, target_w]``."""
        shape = tuple(input_shape)
        if len(shape) < 2:
            raise ValueError(
                f"ResizeFrames expects at least 2 trailing spatial dims, "
                f"got input_shape={shape}"
            )
        return (*shape[:-2], self.height, self.width)

    def __repr__(self) -> str:
        return f"ResizeFrames({self.height}, {self.width})"


class FrameCompose:
    """Sequentially apply a list of frame transforms.

    Chains ``output_shape`` through each transform that exposes it; falls
    back to a dummy-tensor probe for the ones that don't.
    """

    def __init__(self, transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]]):
        self.transforms = list(transforms)
        if not self.transforms:
            raise ValueError("FrameCompose requires at least one transform")

    def __call__(self, frames_uint8: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            frames_uint8 = t(frames_uint8)
        return frames_uint8

    def output_shape(
        self, input_shape: Sequence[int],
    ) -> tuple[int, ...]:
        shape = tuple(input_shape)
        for t in self.transforms:
            shape = _get_output_shape(t, shape)
        return shape

    def __repr__(self) -> str:
        return f"FrameCompose({self.transforms!r})"


def _get_output_shape(
    transform: Callable, input_shape: Sequence[int],
) -> tuple[int, ...]:
    """Derive a transform's output shape from its input shape.

    Prefers the transform's own ``output_shape`` method (cheap, exact);
    falls back to a one-shot probe with a zeroed uint8 tensor — only
    needed for user-supplied transforms that don't advertise their
    output dimensions.
    """
    shape = tuple(input_shape)
    getter = getattr(transform, "output_shape", None)
    if callable(getter):
        out = getter(shape)
        return tuple(int(x) for x in out)
    # Fallback probe: shape-only; uint8 matches the buffer's dtype.
    probe = torch.zeros(shape, dtype=torch.uint8)
    try:
        result = transform(probe)
    except Exception as e:
        raise RuntimeError(
            f"probe of transform {transform!r} failed on input_shape="
            f"{shape}: {type(e).__name__}: {e}.  Either fix the "
            f"transform or give it an output_shape() method."
        ) from e
    if not isinstance(result, torch.Tensor):
        raise RuntimeError(
            f"transform {transform!r} returned non-Tensor "
            f"{type(result).__name__}; expected a uint8 Tensor"
        )
    return tuple(result.shape)


def probe_output_shape(
    transform: Callable | None, input_shape: Sequence[int],
) -> tuple[int, ...]:
    """Public shape-probing helper.

    Returns ``input_shape`` unchanged when ``transform is None``.  Used by
    :meth:`BlendedLeRobotDataModule._setup_shuffle_buffer_training` to
    size the ShuffleBuffer.
    """
    if transform is None:
        return tuple(int(x) for x in input_shape)
    return _get_output_shape(transform, input_shape)
