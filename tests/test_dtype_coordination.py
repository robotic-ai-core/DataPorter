"""Tests for the wire/working dtype coordination layer.

See ``docs/dtype-coordination.md`` for the design rationale this exercises.

The original production-crash repro (fp16 wire + bf16-mixed precision +
``Int8Conv2d`` from utorch) lives at AutoFPV-side
``tmp/repro_int8_conv2d_bf16_autocast.py`` — it isn't included here
because it requires utorch (not a DataPorter dep).  Verifying the
repro after this fix confirms the end-to-end story; the tests in this
file pin the contract DataPorter owns.

The test surface is intentionally focused on observable behavior:

- The coordinator's parsed rules.
- The dtype landed at the model boundary (``apply_working_dtype``).
- The Lightning ``on_after_batch_transfer`` hook end-to-end.

The mixed-precision end-to-end tests skip on CPU-only CI — Lightning's
``16-mixed`` / ``bf16-mixed`` plugins are CUDA-bound.  Make sure CI has
at least one CUDA runner exercising this file or the integration tests
go silent.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
import torch

from dataporter import DtypeCoordinator, PrecisionCoordinationMixin
from dataporter.dtype_coordination import _is_lossy_pair


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakePrecisionPlugin:
    """Stand-in for ``trainer.precision_plugin`` - exposes ``precision`` only."""

    def __init__(self, precision: str):
        self.precision = precision


class _FakeTrainer:
    """Stand-in for the Lightning ``trainer`` attribute on a DataModule."""

    def __init__(self, precision: str):
        self.precision_plugin = _FakePrecisionPlugin(precision)


# ---------------------------------------------------------------------------
# Configuration parsing
# ---------------------------------------------------------------------------


class TestDtypeCoordinatorConfig:
    def test_none_config_is_empty(self):
        c = DtypeCoordinator.from_config(None)
        assert not c.has_rules()
        assert c.wire_converter.dtype_map == {}

    def test_empty_list_is_empty(self):
        c = DtypeCoordinator.from_config([])
        assert not c.has_rules()
        assert c.wire_converter.dtype_map == {}

    def test_legacy_list_form_no_working(self):
        """Old configs (no `working` key) default to `match` -> bug-fix path."""
        c = DtypeCoordinator.from_config([
            {"path": "observation.image", "dtype": "float16"},
        ])
        assert len(c.rules) == 1
        rule = c.rules[0]
        assert rule.path == "observation.image"
        assert rule.wire_dtype == "float16"
        assert rule.working_directive == "match"
        assert c.wire_converter.dtype_map == {"observation.image": "float16"}

    def test_legacy_dict_form(self):
        """Dict form works exactly like list form with implicit working='match'."""
        c = DtypeCoordinator.from_config({"observation.image": "float16"})
        assert len(c.rules) == 1
        assert c.rules[0].working_directive == "match"
        assert c.wire_converter.dtype_map == {"observation.image": "float16"}

    def test_explicit_working_directive(self):
        c = DtypeCoordinator.from_config([
            {"path": "x", "dtype": "float16", "working": "float32"},
            {"path": "y", "dtype": "float16", "working": "keep"},
        ])
        assert c.rules[0].working_directive == "float32"
        assert c.rules[1].working_directive == "keep"

    def test_wire_alias_overrides_dtype(self):
        """``wire`` is the explicit form; if both are set, ``wire`` wins."""
        c = DtypeCoordinator.from_config([
            {"path": "x", "dtype": "float32", "wire": "float16"},
        ])
        assert c.rules[0].wire_dtype == "float16"

    def test_invalid_working_directive_raises_at_construction(self):
        with pytest.raises(ValueError, match="invalid working directive"):
            DtypeCoordinator.from_config([
                {"path": "x", "dtype": "float16", "working": "Float32"},
            ])

    def test_invalid_wire_dtype_raises(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            DtypeCoordinator.from_config([
                {"path": "x", "dtype": "Float32"},
            ])

    def test_missing_path_raises(self):
        with pytest.raises(ValueError, match="must be dicts with at least a 'path'"):
            DtypeCoordinator.from_config([{"dtype": "float16"}])

    def test_typeerror_for_wrong_type(self):
        with pytest.raises(TypeError):
            DtypeCoordinator.from_config("float16")

    def test_has_rules_false_when_all_keep(self):
        c = DtypeCoordinator.from_config([
            {"path": "x", "dtype": "float16", "working": "keep"},
        ])
        assert not c.has_rules()

    def test_normalize_parsed_without_wire(self):
        """A normalize rule needs no explicit wire dtype: the dataset emits
        uint8 directly, so there is nothing to downcast at collate time."""
        c = DtypeCoordinator.from_config([
            {"path": "observation.image", "working": "match", "normalize": 255.0},
        ])
        assert len(c.rules) == 1
        rule = c.rules[0]
        assert rule.normalize == 255.0
        assert rule.wire_dtype is None
        assert rule.working_directive == "match"
        # No wire downcast registered (the dataset already emits uint8).
        assert c.wire_converter.dtype_map == {}
        # A normalize rule always implies a working-side upcast (it must
        # produce a float), so it gates the on_after_batch_transfer walk.
        assert c.has_working_rules()
        assert c.normalize_paths() == ("observation.image",)

    def test_normalize_with_explicit_uint8_wire_ok(self):
        """An explicit integer ``wire: uint8`` is allowed alongside normalize."""
        c = DtypeCoordinator.from_config([
            {"path": "observation.image", "wire": "uint8",
             "working": "match", "normalize": 255.0},
        ])
        assert c.rules[0].normalize == 255.0
        assert c.rules[0].wire_dtype == "uint8"
        # The integer wire IS registered for the collate-time converter.
        assert c.wire_converter.dtype_map == {"observation.image": "uint8"}

    def test_normalize_nonpositive_raises(self):
        with pytest.raises(ValueError, match="'normalize' must be a positive number"):
            DtypeCoordinator.from_config([
                {"path": "x", "working": "match", "normalize": 0},
            ])

    def test_normalize_with_float_wire_raises(self):
        """A floating wire would erase the integer encoding before /scale."""
        with pytest.raises(ValueError, match="'normalize' requires an integer wire"):
            DtypeCoordinator.from_config([
                {"path": "x", "wire": "float16", "working": "match",
                 "normalize": 255.0},
            ])

    def test_normalize_with_keep_raises(self):
        """``working: keep`` can't normalize: the int wire would never upcast."""
        with pytest.raises(ValueError, match="incompatible with working"):
            DtypeCoordinator.from_config([
                {"path": "x", "wire": "uint8", "working": "keep",
                 "normalize": 255.0},
            ])

    def test_normalize_paths_empty_without_normalize(self):
        c = DtypeCoordinator.from_config([
            {"path": "x", "dtype": "float16"},
        ])
        assert c.normalize_paths() == ()


# ---------------------------------------------------------------------------
# Working dtype discovery
# ---------------------------------------------------------------------------


class TestResolveWorkingDtype:
    @pytest.fixture
    def coord(self):
        return DtypeCoordinator.from_config([
            {"path": "x", "dtype": "float16"},
        ])

    @pytest.mark.parametrize("precision,expected", [
        ("bf16-mixed", torch.bfloat16),
        ("16-mixed", torch.float16),
        ("bf16-true", torch.bfloat16),
        ("16-true", torch.float16),
        ("32-true", torch.float32),
        ("64-true", torch.float64),
        # Legacy / numeric forms
        ("16", torch.float16),
        ("bf16", torch.bfloat16),
        ("32", torch.float32),
    ])
    def test_known_precisions(self, coord, precision, expected):
        plugin = _FakePrecisionPlugin(precision)
        assert coord.resolve_working_dtype(plugin) == expected

    def test_none_plugin_returns_none(self, coord):
        assert coord.resolve_working_dtype(None) is None

    def test_plugin_without_precision_returns_none(self, coord):
        class P:
            pass
        assert coord.resolve_working_dtype(P()) is None

    def test_unknown_precision_fails_loudly(self, coord):
        plugin = _FakePrecisionPlugin("future-mixed")
        with pytest.raises(ValueError, match="Unrecognized Lightning precision"):
            coord.resolve_working_dtype(plugin)


# ---------------------------------------------------------------------------
# Apply working dtype
# ---------------------------------------------------------------------------


class TestApplyWorkingDtype:
    def _batch(self, **overrides):
        b = {
            "observation": {
                "image": torch.randn(2, 3, 4, 4, dtype=torch.float16),
                "state": torch.randn(2, 8, dtype=torch.float32),
            },
            "action": torch.randn(2, 4, dtype=torch.float32),
            "metadata": {"episode_index": 0},
        }
        b.update(overrides)
        return b

    def test_no_rules_passes_through(self):
        c = DtypeCoordinator.from_config(None)
        b = self._batch()
        out = c.apply_working_dtype(b, torch.bfloat16)
        assert out["observation"]["image"].dtype == torch.float16

    def test_match_upcasts_to_plugin_dtype(self):
        c = DtypeCoordinator.from_config([
            {"path": "observation.image", "dtype": "float16"},
        ])
        b = self._batch()
        out = c.apply_working_dtype(b, torch.bfloat16)
        assert out["observation"]["image"].dtype == torch.bfloat16
        # Untouched paths preserved.
        assert out["action"].dtype == torch.float32

    def test_match_with_none_working_is_noop(self):
        """Standalone (no Lightning trainer) -> wire dtype survives."""
        c = DtypeCoordinator.from_config([
            {"path": "observation.image", "dtype": "float16"},
        ])
        b = self._batch()
        out = c.apply_working_dtype(b, None)
        assert out["observation"]["image"].dtype == torch.float16

    def test_keep_preserves_wire_dtype(self):
        c = DtypeCoordinator.from_config([
            {"path": "observation.image", "dtype": "float16", "working": "keep"},
        ])
        b = self._batch()
        out = c.apply_working_dtype(b, torch.bfloat16)
        assert out["observation"]["image"].dtype == torch.float16

    def test_explicit_working_overrides_plugin(self):
        c = DtypeCoordinator.from_config([
            {"path": "observation.image", "dtype": "float16", "working": "float32"},
        ])
        b = self._batch()
        out = c.apply_working_dtype(b, torch.bfloat16)
        assert out["observation"]["image"].dtype == torch.float32

    def test_per_key_heterogeneity(self):
        """Different rules per path can express any combination."""
        c = DtypeCoordinator.from_config([
            {"path": "observation.image", "dtype": "float16"},  # match -> bf16
            {"path": "action", "dtype": "float32", "working": "keep"},  # stays
            {"path": "observation.state", "dtype": "float32",
             "working": "float16"},  # explicit
        ])
        b = self._batch()
        out = c.apply_working_dtype(b, torch.bfloat16)
        assert out["observation"]["image"].dtype == torch.bfloat16
        assert out["observation"]["state"].dtype == torch.float16
        assert out["action"].dtype == torch.float32

    def test_nonfloat_tensor_ignores_working(self):
        c = DtypeCoordinator.from_config([
            {"path": "ids", "dtype": "int32"},
        ])
        b = {"ids": torch.tensor([1, 2, 3], dtype=torch.int32)}
        out = c.apply_working_dtype(b, torch.bfloat16)
        assert out["ids"].dtype == torch.int32

    def test_nonfloat_tensor_with_explicit_working_target_is_upcast(self):
        """uint8 wire pixels + explicit ``working: bfloat16`` upcasts on GPU.

        Mirrors the workflow where a user compresses image data to uint8
        on the wire to halve worker-buffer RAM and PCIe transfer cost,
        then relies on DataPorter to upcast at the model boundary.  The
        default ``match`` directive stays conservative (preserves the
        token-id case in
        :py:meth:`test_nonfloat_tensor_ignores_working`); an explicit
        target dtype is an intentional opt-in.
        """
        c = DtypeCoordinator.from_config([
            {
                "path": "observation.image",
                "dtype": "uint8",
                "working": "bfloat16",
            },
        ])
        b = {
            "observation": {
                "image": torch.randint(0, 256, (2, 3, 4, 4), dtype=torch.uint8),
            }
        }
        out = c.apply_working_dtype(b, torch.float16)
        # Honored explicit target, NOT the precision-plugin dtype.
        assert out["observation"]["image"].dtype == torch.bfloat16

    def test_nonfloat_tensor_with_keep_directive_stays_nonfloat(self):
        """``working: keep`` short-circuits the upcast even for non-float
        wires — the safety hatch users can reach for if the new
        explicit-target behaviour ever surprises them.
        """
        c = DtypeCoordinator.from_config([
            {"path": "ids", "dtype": "int32", "working": "keep"},
        ])
        b = {"ids": torch.tensor([1, 2, 3], dtype=torch.int32)}
        out = c.apply_working_dtype(b, torch.bfloat16)
        assert out["ids"].dtype == torch.int32

    def test_path_not_in_batch_is_silent(self):
        c = DtypeCoordinator.from_config([
            {"path": "missing.path", "dtype": "float16"},
        ])
        b = self._batch()
        out = c.apply_working_dtype(b, torch.bfloat16)
        # Just shouldn't error; observation.image stays fp16.
        assert out["observation"]["image"].dtype == torch.float16

    def test_already_target_dtype_short_circuits(self):
        """No conversion when wire == working (idempotent)."""
        c = DtypeCoordinator.from_config([
            {"path": "x", "dtype": "float16"},
        ])
        b = {"x": torch.randn(4, dtype=torch.float16)}
        out = c.apply_working_dtype(b, torch.float16)
        assert out["x"].dtype == torch.float16
        # We don't check identity (.to() may copy) but dtype is preserved.

    def test_list_and_tuple_traversal(self):
        c = DtypeCoordinator.from_config([
            {"path": "frames[0]", "dtype": "float16"},
        ])
        b = {"frames": [torch.randn(4, dtype=torch.float16)]}
        out = c.apply_working_dtype(b, torch.bfloat16)
        assert out["frames"][0].dtype == torch.bfloat16

    def test_lossy_round_trip_warns_once(self, caplog):
        c = DtypeCoordinator.from_config([
            {"path": "x", "dtype": "float16"},
        ])
        b = {"x": torch.randn(4, dtype=torch.float16)}
        with caplog.at_level(logging.WARNING, logger="dataporter.dtype_coordination"):
            c.apply_working_dtype(b, torch.bfloat16)
            c.apply_working_dtype(b, torch.bfloat16)  # second call: no new warn
        warnings = [r for r in caplog.records if "lossy round-trip" in r.message]
        assert len(warnings) == 1


# ---------------------------------------------------------------------------
# Lossy-pair classification
# ---------------------------------------------------------------------------


class TestLossyPair:
    @pytest.mark.parametrize("wire,working,expected", [
        (torch.float16, torch.bfloat16, True),
        (torch.bfloat16, torch.float16, True),
        (torch.float64, torch.float16, True),
        (torch.float64, torch.bfloat16, True),
        (torch.float32, torch.float16, False),  # standard mixed-precision
        (torch.float16, torch.float32, False),  # the fix path
        (torch.bfloat16, torch.float32, False),
        (torch.int32, torch.float16, False),  # non-float
    ])
    def test_classification(self, wire, working, expected):
        assert _is_lossy_pair(wire, working) is expected


# ---------------------------------------------------------------------------
# PrecisionCoordinationMixin (Lightning hook)
# ---------------------------------------------------------------------------


class _MockDataModule(PrecisionCoordinationMixin):
    """Bare mixin host - we don't import Lightning here.

    The mixin's ``on_after_batch_transfer`` only touches
    ``self.dtype_coordinator`` and ``self.trainer``, so we can exercise
    it without the full LightningDataModule machinery.
    """

    def __init__(self, dtype_conversions=None, trainer=None):
        self._init_precision_coordination(dtype_conversions)
        self.trainer = trainer


class TestPrecisionCoordinationMixin:
    def test_passes_through_when_no_rules(self):
        dm = _MockDataModule(dtype_conversions=None,
                             trainer=_FakeTrainer("bf16-mixed"))
        b = {"x": torch.randn(4, dtype=torch.float32)}
        out = dm.on_after_batch_transfer(b, dataloader_idx=0)
        assert out["x"].dtype == torch.float32

    def test_upcasts_with_bf16_mixed(self):
        dm = _MockDataModule(
            dtype_conversions=[{"path": "image", "dtype": "float16"}],
            trainer=_FakeTrainer("bf16-mixed"),
        )
        b = {"image": torch.randn(2, 3, 4, 4, dtype=torch.float16)}
        out = dm.on_after_batch_transfer(b, dataloader_idx=0)
        assert out["image"].dtype == torch.bfloat16

    def test_upcasts_with_16_mixed_is_noop(self):
        dm = _MockDataModule(
            dtype_conversions=[{"path": "image", "dtype": "float16"}],
            trainer=_FakeTrainer("16-mixed"),
        )
        b = {"image": torch.randn(2, 3, 4, 4, dtype=torch.float16)}
        out = dm.on_after_batch_transfer(b, dataloader_idx=0)
        assert out["image"].dtype == torch.float16

    def test_upcasts_with_32_true(self):
        dm = _MockDataModule(
            dtype_conversions=[{"path": "image", "dtype": "float16"}],
            trainer=_FakeTrainer("32-true"),
        )
        b = {"image": torch.randn(2, 3, 4, 4, dtype=torch.float16)}
        out = dm.on_after_batch_transfer(b, dataloader_idx=0)
        assert out["image"].dtype == torch.float32

    def test_no_trainer_is_noop(self):
        """Standalone use (datamodule constructed without a Trainer)."""
        dm = _MockDataModule(
            dtype_conversions=[{"path": "image", "dtype": "float16"}],
            trainer=None,
        )
        b = {"image": torch.randn(2, 3, 4, 4, dtype=torch.float16)}
        out = dm.on_after_batch_transfer(b, dataloader_idx=0)
        assert out["image"].dtype == torch.float16


# ---------------------------------------------------------------------------
# End-to-end: Lightning DataModule + Trainer
# ---------------------------------------------------------------------------


def _build_lightning_dm(dtype_conversions):
    """Build a minimal LightningDataModule that uses the mixin.

    Imports Lightning lazily so the rest of the file works without it.
    """
    import lightning as L
    from torch.utils.data import DataLoader, Dataset

    class _DS(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, _idx):
            return {
                "image": torch.rand(3, 8, 8, dtype=torch.float16),
                "action": torch.randn(2, dtype=torch.float32),
            }

    class _DM(PrecisionCoordinationMixin, L.LightningDataModule):
        def __init__(self):
            super().__init__()
            self._init_precision_coordination(dtype_conversions)

        def train_dataloader(self):
            return DataLoader(_DS(), batch_size=2)

        def val_dataloader(self):
            return DataLoader(_DS(), batch_size=2)

    return _DM()


class _Probe:
    """Capture the dtype of ``image`` as the model sees it."""

    def __init__(self):
        self.image_dtype: torch.dtype | None = None
        self.action_dtype: torch.dtype | None = None


def _build_lightning_module(probe: _Probe):
    import lightning as L

    class _LM(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 2)

        def training_step(self, batch, batch_idx):
            probe.image_dtype = batch["image"].dtype
            probe.action_dtype = batch["action"].dtype
            return self.layer(batch["action"]).sum()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=1e-3)

    return _LM()


@pytest.mark.parametrize("precision,expected_image_dtype", [
    ("32-true", torch.float32),
    ("bf16-mixed", torch.bfloat16),
    ("16-mixed", torch.float16),
])
def test_end_to_end_lightning_hook(precision, expected_image_dtype):
    """The full Lightning pipeline routes wire fp16 -> working as advertised.

    This is the integration test for the bug Neil reported: a Lightning
    Trainer with the named precision should see the upcast happen
    automatically when ``PrecisionCoordinationMixin`` is applied.
    """
    pytest.importorskip("lightning")
    import lightning as L

    if precision in ("bf16-mixed", "16-mixed") and not torch.cuda.is_available():
        # Lightning's mixed precision requires CUDA; CPU plugins exist but
        # autocast on CPU has different dtype semantics. We skip rather
        # than spuriously assert.
        pytest.skip("mixed precision requires CUDA")

    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    dm = _build_lightning_dm(
        dtype_conversions=[{"path": "image", "dtype": "float16"}],
    )
    probe = _Probe()
    model = _build_lightning_module(probe)
    trainer = L.Trainer(
        max_steps=1,
        limit_val_batches=0,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)
    assert probe.image_dtype == expected_image_dtype
    # action wasn't in dtype_conversions, so it should keep whatever
    # device-transfer-time dtype Lightning yields (fp32 in all these cases).
    assert probe.action_dtype == torch.float32


def test_end_to_end_legacy_config_no_working_field():
    """Configs without ``working`` get the bug-fix behavior automatically."""
    pytest.importorskip("lightning")
    if not torch.cuda.is_available():
        pytest.skip("bf16-mixed requires CUDA")

    import lightning as L

    dm = _build_lightning_dm(
        dtype_conversions=[{"path": "image", "dtype": "float16"}],
    )
    probe = _Probe()
    model = _build_lightning_module(probe)
    trainer = L.Trainer(
        max_steps=1,
        limit_val_batches=0,
        accelerator="cuda",
        devices=1,
        precision="bf16-mixed",
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)
    # Legacy config (no working field) gets implicit working='match' -> bf16.
    assert probe.image_dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# uint8 wire: normalize-on-upcast (the 4x-smaller-wire optimization)
# ---------------------------------------------------------------------------


class TestNormalizeWire:
    """The ``normalize`` directive: a uint8 ``[0, 255]`` wire tensor is upcast
    to the working float dtype AND divided by the scale on the GPU side, so
    the model sees the same ``[0, 1]`` floats a float wire would have carried
    — at 1/4 the bytes.  See ``normalize_paths`` + ``_maybe_upcast``.
    """

    def _uint8_batch(self):
        return {
            "observation": {
                "image": torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8),
            }
        }

    def test_float32_normalize_is_bit_exact_to_dataset_path(self):
        """uint8 -> float32 / 255 via the coordinator is BIT-IDENTICAL to the
        dataset's own ``frames.to(float32) / 255`` — the parity anchor that
        lets us move the /255 off the dataloader without changing a single
        model input.
        """
        c = DtypeCoordinator.from_config([
            {"path": "observation.image", "working": "float32", "normalize": 255.0},
        ])
        b = self._uint8_batch()
        raw = b["observation"]["image"]
        expected = raw.to(torch.float32) / 255.0
        out = c.apply_working_dtype(b, torch.float32)
        got = out["observation"]["image"]
        assert got.dtype == torch.float32
        assert torch.equal(got, expected)

    def test_bf16_normalize_matches_within_tolerance(self):
        """Under a bf16 trainer the normalized result matches the float32
        reference to bf16 precision.  bf16 has 8 mantissa bits, so the
        worst-case rounding error on a ``[0, 1]`` value is below 2^-7.
        """
        c = DtypeCoordinator.from_config([
            {"path": "observation.image", "working": "match", "normalize": 255.0},
        ])
        b = self._uint8_batch()
        reference = b["observation"]["image"].to(torch.float32) / 255.0
        out = c.apply_working_dtype(b, torch.bfloat16)
        got = out["observation"]["image"]
        assert got.dtype == torch.bfloat16
        assert got.to(torch.float32).max() <= 1.0
        torch.testing.assert_close(
            got.to(torch.float32), reference, atol=2 ** -7, rtol=0
        )

    def test_normalize_without_working_dtype_raises(self):
        """A ``match`` normalize rule with no resolved working dtype (no
        Lightning trainer) can't pick a float to divide in — fail loudly
        rather than silently emit raw uint8 ints or the wrong scale.
        """
        c = DtypeCoordinator.from_config([
            {"path": "observation.image", "working": "match", "normalize": 255.0},
        ])
        b = self._uint8_batch()
        with pytest.raises(
            ValueError, match="'normalize' needs a resolved working float"
        ):
            c.apply_working_dtype(b, None)

    def test_normalize_skips_already_float_tensor(self):
        """If a path already carries a float tensor, the normalize branch is
        skipped (no spurious /255) — only an integer wire is normalized.
        """
        c = DtypeCoordinator.from_config([
            {"path": "observation.image", "working": "float32", "normalize": 255.0},
        ])
        already_float = torch.rand(2, 3, 8, 8, dtype=torch.float32)
        b = {"observation": {"image": already_float}}
        out = c.apply_working_dtype(b, torch.float32)
        assert torch.equal(out["observation"]["image"], already_float)

    def test_mixin_normalizes_uint8_under_bf16_trainer(self):
        """End-to-end through the Lightning hook: a uint8 batch under a
        bf16-mixed trainer lands as normalized bf16 at the model boundary.
        """
        dm = _MockDataModule(
            dtype_conversions=[
                {"path": "observation.image", "working": "match",
                 "normalize": 255.0},
            ],
            trainer=_FakeTrainer("bf16-mixed"),
        )
        b = {
            "observation": {
                "image": torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8),
            }
        }
        reference = b["observation"]["image"].to(torch.float32) / 255.0
        out = dm.on_after_batch_transfer(b, dataloader_idx=0)
        got = out["observation"]["image"]
        assert got.dtype == torch.bfloat16
        assert got.to(torch.float32).max() <= 1.0
        torch.testing.assert_close(
            got.to(torch.float32), reference, atol=2 ** -7, rtol=0
        )
