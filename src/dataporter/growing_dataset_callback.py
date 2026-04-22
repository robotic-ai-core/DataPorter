"""Lightning Callback that drives ``refresh()`` at each epoch boundary.

Users wire a growing dataset to Lightning training with one line::

    from dataporter import GrowingDatasetCallback
    trainer = Trainer(callbacks=[GrowingDatasetCallback()])

On every ``on_train_epoch_start``, the callback walks the trainer's
DataModule and calls ``.refresh()`` on any attribute named
``train_dataset`` that exposes the method.  That's it — no hooks on the
DataModule itself, no ``reload_dataloaders_every_n_epochs`` flag
required.  The Dataset stays framework-agnostic; the callback is the
Lightning-specific glue.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class GrowingDatasetCallback:
    """Calls ``datamodule.train_dataset.refresh()`` at epoch start.

    Compatible with pytorch-lightning's ``Callback`` interface via duck
    typing — it exposes ``on_train_epoch_start(trainer, pl_module)`` and
    inherits from ``pl.Callback`` when Lightning is importable.  When
    Lightning isn't installed, it still imports cleanly; it just isn't
    useful.

    The callback is a **no-op** if:

    - The trainer has no ``datamodule`` attribute.
    - The datamodule has no ``train_dataset`` attribute.
    - The dataset doesn't expose a ``refresh`` method.

    This intentionally tolerates static datasets, so adding the callback
    to a Trainer that doesn't need it is harmless.
    """

    def __init__(self):
        # Try to inherit from Lightning's Callback when available so
        # downstream tooling (type checks, callback dispatch) works the
        # standard way.  Fall back to a bare class if lightning isn't
        # importable.
        pass

    def on_train_epoch_start(self, trainer, pl_module) -> None:
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


# Try to make GrowingDatasetCallback a real Lightning Callback subclass
# so ``isinstance(cb, pl.Callback)`` works and Lightning's callback
# dispatch treats it natively.  Degrades to the plain class above if
# Lightning isn't installed.
try:
    import lightning.pytorch as _pl

    class GrowingDatasetCallback(_pl.Callback):   # type: ignore[no-redef]
        """See module docstring."""
        __doc__ = GrowingDatasetCallback.__doc__

        def on_train_epoch_start(self, trainer, pl_module) -> None:
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
                    f"{type(e).__name__}: {e}.  Training continues with "
                    f"the previous admitted set."
                )
except Exception:
    # Lightning not importable; keep the plain class above.
    pass
