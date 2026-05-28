"""Wire/working dtype coordination.

DataPorter's ``dtype_conversions`` performs a bandwidth-optimizing
**wire-dtype** downcast at collate time (typically pixels fp32 -> fp16).
Lightning's precision plugin sets the **working dtype** the model should
see (e.g. ``"bf16-mixed"`` -> bf16 inside autocast).  When wire and
working disagree (fp16 wire vs bf16 plugin), non-autocastable layers
crash with ``Input/bias dtype`` mismatches.

This module bridges them.  See ``docs/dtype-coordination.md`` for the
full design rationale.

Public API:

- :class:`DtypeCoordinator` - parses the rules, resolves working dtypes
  from a precision plugin, and applies the upcast to a batch.
- :class:`PrecisionCoordinationMixin` - drop into a
  :class:`lightning.LightningDataModule` to wire the upcast to
  ``on_after_batch_transfer``.

Both layers are deliberately decoupled so the coordinator can be used
outside Lightning (tests, scripts, ad-hoc inference) without dragging in
the Lightning import.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch

from .converters import TORCH_DTYPE_NAMES, KeyBasedDtypeConverter

logger = logging.getLogger(__name__)


# Sentinel values for the per-rule ``working`` field.  They are strings so
# YAML configs stay declarative.  Any other string must resolve to a torch
# floating dtype via ``KeyBasedDtypeConverter.torch_dtypes``.
_WORKING_MATCH = "match"  # Match the precision plugin's working dtype.
_WORKING_KEEP = "keep"  # Keep the wire dtype - do not upcast.

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)

# Map from Lightning's precision-plugin string to the working dtype.
# Source: ``trainer.precision_plugin.precision`` (public attribute).
_PRECISION_TO_DTYPE: Dict[str, torch.dtype] = {
    "bf16-mixed": torch.bfloat16,
    "16-mixed": torch.float16,
    "bf16-true": torch.bfloat16,
    "16-true": torch.float16,
    "32-true": torch.float32,
    "64-true": torch.float64,
    # Legacy / numeric forms occasionally surfaced by older Lightning APIs.
    "16": torch.float16,
    "bf16": torch.bfloat16,
    "32": torch.float32,
    "64": torch.float64,
}


def _string_to_torch_dtype(name: str) -> torch.dtype:
    """Resolve a dtype string against the canonical name → ``torch.dtype``
    table.

    Single source of truth: :data:`dataporter.converters.TORCH_DTYPE_NAMES`.
    Keeps the wire and working halves of the coordinator in lockstep with
    :class:`KeyBasedDtypeConverter`.
    """
    if name not in TORCH_DTYPE_NAMES:
        raise ValueError(
            f"Unsupported dtype string {name!r}. "
            f"Supported: {sorted(TORCH_DTYPE_NAMES.keys())}"
        )
    return TORCH_DTYPE_NAMES[name]


@dataclass(frozen=True)
class _Rule:
    """Compiled per-path coordination rule.

    Attributes:
        path: Dotted batch path the rule matches.
        wire_dtype: Target dtype at collate time (None = don't downcast).
        working_directive: One of ``"match"``, ``"keep"``, or an explicit
            dtype string. Resolved at runtime against the precision plugin.
    """

    path: str
    wire_dtype: Optional[str]
    working_directive: str


@dataclass
class DtypeCoordinator:
    """Owns the wire-and-working dtype coordination for a datamodule.

    The coordinator exposes two halves:

    - ``wire_converter``: the existing :class:`KeyBasedDtypeConverter`,
      built from the ``dtype`` field of each rule. Pass this to
      :class:`ResumableDataLoader` (via ``converter=``) so collate-time
      downcast happens in workers.
    - :meth:`apply_working_dtype`: walks a batch tree (already on-device)
      and upcasts each declared path according to its ``working`` directive.

    Construct via :meth:`from_config` for the YAML-friendly path.
    """

    rules: Tuple[_Rule, ...]
    wire_converter: KeyBasedDtypeConverter
    # O(1) path → rule lookup, populated in __post_init__.  Avoids a
    # linear scan on every leaf tensor in :meth:`apply_working_dtype`.
    _rule_by_path: Dict[str, _Rule] = field(default_factory=dict, repr=False)
    # Memoize lossy-round-trip warnings so we emit them once per
    # (wire, working, path) triple instead of every batch.
    _warned_pairs: set = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        # Build the lookup dict once; rules are immutable after
        # construction so this stays correct.
        self._rule_by_path = {r.path: r for r in self.rules}

    @classmethod
    def from_config(
        cls,
        dtype_conversions: Union[List[Dict[str, str]], Dict[str, str], None],
    ) -> "DtypeCoordinator":
        """Parse a YAML-style ``dtype_conversions`` config into a coordinator.

        Accepts:

        - ``None`` / ``[]`` / ``{}`` -> empty coordinator (no-op).
        - List form: ``[{"path": ..., "dtype": ..., "working": ...}, ...]``.
          ``working`` is optional, defaults to ``"match"``.
        - Dict form (legacy): ``{"path": "dtype_str"}``. ``working`` defaults
          to ``"match"`` for every entry. There is no way to set ``working``
          per-key in the dict form - if you need that, switch to list form.
        """
        rules: List[_Rule] = []
        wire_pairs: List[Dict[str, str]] = []

        if dtype_conversions is None:
            pass
        elif isinstance(dtype_conversions, list):
            for entry in dtype_conversions:
                if not isinstance(entry, dict) or "path" not in entry:
                    raise ValueError(
                        "List-form dtype_conversions entries must be dicts "
                        f"with at least a 'path' key. Got: {entry!r}"
                    )
                path = entry["path"]
                # Backward compat: ``dtype`` is the wire dtype. Either
                # ``dtype`` or ``wire`` is accepted; ``wire`` wins if both
                # are set so a user can be explicit.
                wire = entry.get("wire", entry.get("dtype"))
                working = entry.get("working", _WORKING_MATCH)
                rules.append(
                    _Rule(
                        path=path,
                        wire_dtype=wire,
                        working_directive=working,
                    )
                )
                if wire is not None:
                    wire_pairs.append({"path": path, "dtype": wire})
        elif isinstance(dtype_conversions, dict):
            for path, wire in dtype_conversions.items():
                rules.append(
                    _Rule(
                        path=path,
                        wire_dtype=wire,
                        working_directive=_WORKING_MATCH,
                    )
                )
                wire_pairs.append({"path": path, "dtype": wire})
        else:
            raise TypeError(
                "dtype_conversions must be a list of dicts, a dict, or None. "
                f"Got: {type(dtype_conversions).__name__}"
            )

        # Validate working directives up front so config errors fail at
        # construction time, not on the first training batch.
        for rule in rules:
            cls._validate_working_directive(rule)

        wire_converter = KeyBasedDtypeConverter(wire_pairs if wire_pairs else None)
        return cls(rules=tuple(rules), wire_converter=wire_converter)

    @staticmethod
    def _validate_working_directive(rule: _Rule) -> None:
        directive = rule.working_directive
        if directive in (_WORKING_MATCH, _WORKING_KEEP):
            return
        # Otherwise must be a known dtype string - resolve once to surface
        # typos (e.g. ``"Float32"``) before any batch touches the rule.
        try:
            _string_to_torch_dtype(directive)
        except ValueError as e:
            raise ValueError(
                f"dtype_conversions entry for path {rule.path!r}: invalid "
                f"working directive {directive!r}. Allowed: {_WORKING_MATCH!r}, "
                f"{_WORKING_KEEP!r}, or a dtype string. ({e})"
            ) from None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def resolve_working_dtype(
        self, precision_plugin: Any
    ) -> Optional[torch.dtype]:
        """Read the working dtype off a Lightning precision plugin.

        Args:
            precision_plugin: ``trainer.precision_plugin`` or any object
                exposing a ``precision`` string attribute. ``None`` is
                treated as "no Lightning context" -> return ``None``.

        Returns:
            The torch dtype that the model expects to see, or ``None``
            when no Lightning context is available (standalone use).

        Raises:
            ValueError: when ``precision_plugin.precision`` is a string
                we don't recognize. We fail loudly so a future Lightning
                version doesn't silently misroute precision.
        """
        if precision_plugin is None:
            return None
        precision = getattr(precision_plugin, "precision", None)
        if precision is None:
            return None
        # Lightning sometimes uses int / numpy ints as precision strings;
        # normalize to str.
        precision_str = str(precision)
        if precision_str not in _PRECISION_TO_DTYPE:
            raise ValueError(
                f"Unrecognized Lightning precision {precision_str!r}. "
                f"Known values: {sorted(_PRECISION_TO_DTYPE.keys())}. "
                f"If a newer Lightning version added this mode, the "
                f"DtypeCoordinator mapping in dataporter.dtype_coordination "
                f"needs updating."
            )
        return _PRECISION_TO_DTYPE[precision_str]

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def has_working_rules(self) -> bool:
        """True if any rule needs a working-dtype upcast.

        Rules with ``working: 'keep'`` only own the wire-side downcast
        (collate-time, via ``wire_converter``); they don't trigger any
        post-device-transfer work.  This predicate gates the
        ``on_after_batch_transfer`` walk.
        """
        return any(r.working_directive != _WORKING_KEEP for r in self.rules)

    # Back-compat alias — ``has_rules`` was the original name.  Keep it
    # working so external callers and the older test surface don't
    # break, but prefer ``has_working_rules`` in new code.
    has_rules = has_working_rules

    def apply_working_dtype(
        self,
        batch: Any,
        working_dtype: Optional[torch.dtype],
    ) -> Any:
        """Walk ``batch`` and upcast each path-matching tensor.

        Args:
            batch: Nested dict/list/tuple of tensors. Leaves that aren't
                tensors are passed through unchanged.
            working_dtype: The dtype to use when a rule says
                ``working: "match"``. ``None`` -> ``"match"`` becomes a
                no-op (standalone use, no Lightning trainer attached).

        Returns:
            Batch with the same structure; tensors at matching paths are
            upcast in-place when their dtype differs from the resolved
            target. Other tensors are passed through unchanged.
        """
        if not self.rules:
            return batch
        return self._walk(batch, "", working_dtype)

    # Internal helpers -------------------------------------------------

    def _resolve_target(
        self,
        rule: _Rule,
        working_dtype: Optional[torch.dtype],
    ) -> Optional[torch.dtype]:
        """Compute the torch dtype this rule wants the tensor to land at."""
        directive = rule.working_directive
        if directive == _WORKING_KEEP:
            return None  # signal: do not touch
        if directive == _WORKING_MATCH:
            return working_dtype  # may be None (standalone -> no-op)
        return _string_to_torch_dtype(directive)

    def _walk(
        self,
        obj: Any,
        path: str,
        working_dtype: Optional[torch.dtype],
    ) -> Any:
        if isinstance(obj, dict):
            # Lightning's batch_to_device returns dicts (not OrderedDict
            # specifically); preserve the type to avoid surprise.
            return type(obj)({
                k: self._walk(v, _join(path, k), working_dtype)
                for k, v in obj.items()
            })
        if isinstance(obj, (list, tuple)):
            converted = [
                self._walk(v, f"{path}[{i}]", working_dtype)
                for i, v in enumerate(obj)
            ]
            return type(obj)(converted)
        if isinstance(obj, torch.Tensor):
            return self._maybe_upcast(obj, path, working_dtype)
        return obj

    def _maybe_upcast(
        self,
        tensor: torch.Tensor,
        path: str,
        working_dtype: Optional[torch.dtype],
    ) -> torch.Tensor:
        rule = self._lookup_rule(path)
        if rule is None:
            return tensor
        target = self._resolve_target(rule, working_dtype)
        if target is None:
            return tensor
        # Non-floating sources (int/bool) are usually categorical labels
        # (token IDs, attention masks).  Silently autocasting them to a
        # float would break embedding lookups downstream.  When the
        # user only said ``working: "match"`` we stay conservative and
        # skip the cast.  An EXPLICIT target dtype (e.g.
        # ``working: "bfloat16"``) is an intentional opt-in — honor it,
        # so workflows that compress pixels to uint8 on the wire and
        # need them upcast on the GPU keep working without a model-
        # side cast.
        if (
            not tensor.is_floating_point()
            and rule.working_directive == _WORKING_MATCH
        ):
            return tensor
        if tensor.dtype == target:
            return tensor
        # Lossy round-trip warning. Emit once per (wire, working, path).
        if rule.wire_dtype is not None:
            self._maybe_warn_lossy(rule.wire_dtype, target, path)
        return tensor.to(target)

    def _lookup_rule(self, path: str) -> Optional[_Rule]:
        return self._rule_by_path.get(path)

    def _maybe_warn_lossy(
        self,
        wire_dtype_str: str,
        working_dtype: torch.dtype,
        path: str,
    ) -> None:
        try:
            wire_dtype = _string_to_torch_dtype(wire_dtype_str)
        except ValueError:
            return
        key = (wire_dtype, working_dtype, path)
        if key in self._warned_pairs:
            return
        if _is_lossy_pair(wire_dtype, working_dtype):
            logger.warning(
                "Dtype coordination at %r: wire=%s -> working=%s is a "
                "lossy round-trip. Consider matching wire and working, or "
                "use working: 'keep' to skip the upcast.",
                path,
                wire_dtype,
                working_dtype,
            )
            self._warned_pairs.add(key)


def _join(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


def _is_lossy_pair(wire: torch.dtype, working: torch.dtype) -> bool:
    """Return True if going wire -> working is information-losing.

    fp16 <-> bf16 is the canonical lossy pair (different exponent ranges).
    fp64 -> fp16/bf16 also drops a lot of dynamic range.  fp32 -> fp16
    can saturate but is the standard mixed-precision flow, so we don't
    flag it (would be too noisy).
    """
    if wire not in _FLOAT_DTYPES or working not in _FLOAT_DTYPES:
        return False
    if {wire, working} == {torch.float16, torch.bfloat16}:
        return True
    if wire == torch.float64 and working in (torch.float16, torch.bfloat16):
        return True
    return False


# ---------------------------------------------------------------------------
# Lightning glue
# ---------------------------------------------------------------------------


class PrecisionCoordinationMixin:
    """Mixin for ``L.LightningDataModule`` that wires the upcast.

    Usage::

        from dataporter import PrecisionCoordinationMixin
        import lightning as L

        class MyDM(PrecisionCoordinationMixin, L.LightningDataModule):
            def __init__(self, ...):
                super().__init__()
                self._init_precision_coordination(
                    dtype_conversions=[
                        {"path": "observation.image", "dtype": "float16"},
                    ],
                )

    The mixin does **not** apply the wire downcast - that still happens
    in the dataloader's collate fn via :class:`ResumableDataLoader` or by
    passing ``self.dtype_coordinator.wire_converter`` explicitly.

    What the mixin owns:

    - ``self.dtype_coordinator``: the parsed coordinator (also exposes
      ``wire_converter`` for setting up the dataloader).
    - ``on_after_batch_transfer``: the post-device-move upcast.

    Idempotency: if a subclass overrides ``on_after_batch_transfer``,
    chain via ``super().on_after_batch_transfer(batch, dataloader_idx)``
    to keep the upcast in place.
    """

    def _init_precision_coordination(
        self,
        dtype_conversions: Union[List[Dict[str, str]], Dict[str, str], None],
    ) -> None:
        """Build the coordinator. Call from ``__init__`` after ``super().__init__()``."""
        self.dtype_coordinator = DtypeCoordinator.from_config(dtype_conversions)

    def on_after_batch_transfer(
        self, batch: Any, dataloader_idx: int = 0
    ) -> Any:
        """Apply working-dtype upcast after the device transfer.

        Lightning calls this with the batch already on the target device.
        We resolve the working dtype from the precision plugin and walk
        the batch tree.

        When called outside a Lightning trainer (rare but supported -
        e.g. the user manually invoked ``transfer_batch_to_device``),
        ``self.trainer`` is ``None`` and the coordinator falls back to
        a no-op upcast.
        """
        coord: Optional[DtypeCoordinator] = getattr(self, "dtype_coordinator", None)
        if coord is None or not coord.has_working_rules():
            return batch
        precision_plugin = None
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            precision_plugin = getattr(trainer, "precision_plugin", None)
        working_dtype = coord.resolve_working_dtype(precision_plugin)
        return coord.apply_working_dtype(batch, working_dtype)
