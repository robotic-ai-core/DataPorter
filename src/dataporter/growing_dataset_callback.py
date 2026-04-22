"""Lightning Callback that drives ``refresh()`` on the training dataset.

Users wire a growing dataset to Lightning training with one line::

    from dataporter import GrowingDatasetCallback
    trainer = Trainer(callbacks=[GrowingDatasetCallback()])

Default: refreshes on ``on_train_epoch_start``.  For ``max_steps``-driven
training on large datasets where epoch boundaries are too coarse, pass
``every_n_steps=N`` to refresh every N training steps instead::

    trainer = Trainer(
        callbacks=[GrowingDatasetCallback(every_n_steps=500)],
        max_steps=10_000,
    )

The callback walks the trainer's DataModule and calls ``.refresh()`` on
any attribute named ``train_dataset`` that exposes the method.  The
Dataset stays framework-agnostic; the callback is the Lightning-specific
glue.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _noop_callback_base():
    """Fallback base class when Lightning isn't importable."""

    class _NoopCallback:
        pass

    return _NoopCallback


try:
    import lightning.pytorch as _pl
    _Callback = _pl.Callback
except Exception:
    _Callback = _noop_callback_base()


class GrowingDatasetCallback(_Callback):  # type: ignore[misc,valid-type]
    """Calls ``datamodule.train_dataset.refresh()`` at a configurable cadence.

    Two modes:

    - ``every_n_steps=None`` (default): refresh on ``on_train_epoch_start``.
      Natural rhythm — Lightning fires epoch-start whenever the DataLoader
      cycles, so a small admitted set auto-refreshes frequently until it
      grows.
    - ``every_n_steps=N``: refresh on ``on_train_batch_start`` whenever
      ``trainer.global_step`` is a positive multiple of N.  Use with
      ``max_steps`` training on large datasets where epochs have grown
      long enough that step-level granularity gets newly-downloaded
      episodes into training faster.

    No-op if:

    - The trainer has no ``datamodule`` attribute.
    - The datamodule has no ``train_dataset``.
    - The dataset doesn't expose a ``refresh`` method.

    Harmless to add to a Trainer that doesn't need it.
    """

    def __init__(self, every_n_steps: int | None = None):
        super().__init__()
        if every_n_steps is not None and every_n_steps <= 0:
            raise ValueError(
                f"every_n_steps must be positive, got {every_n_steps!r}"
            )
        self.every_n_steps = every_n_steps

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if self.every_n_steps is None:
            self._refresh(trainer)

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx,
    ) -> None:
        if self.every_n_steps is None:
            return
        step = getattr(trainer, "global_step", None)
        if step is None or step <= 0:
            return
        if step % self.every_n_steps == 0:
            self._refresh(trainer)

    # ------------------------------------------------------------------
    # Shared refresh mechanics
    # ------------------------------------------------------------------

    def _refresh(self, trainer) -> None:
        dm = getattr(trainer, "datamodule", None)
        if dm is None:
            return
        ds = getattr(dm, "train_dataset", None)
        if ds is None:
            return
        refresh = getattr(ds, "refresh", None)
        if not callable(refresh):
            return
        try:
            refresh()
        except Exception as e:
            logger.warning(
                f"GrowingDatasetCallback: refresh() raised "
                f"{type(e).__name__}: {e}.  Training continues with the "
                f"previous admitted set."
            )
