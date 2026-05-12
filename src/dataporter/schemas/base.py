"""Generic Schema framework for batch/sample validation.

A :class:`Schema` declares per-field invariants (:class:`FieldSpec`) and
optional per-source extra checks.  Subclasses pick whether they validate
per-sample (text-style, pre-collate) or per-batch (LeRobot-style,
post-collate) by how they describe ``FIELDS`` shapes.

Per-sample contracts (e.g. ``TextSampleSpec``) declare shapes without a
leading batch dimension and are typically validated against ``Dataset``
items.  Per-batch contracts (e.g. ``VideoActionBatchSpec``) include a
leading wildcard batch dim and are validated against ``DataLoader``
output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Iterable, Iterator, Mapping

import torch
from torch.utils.data import Dataset


class SchemaError(ValueError):
    """Raised when data violates a Schema contract."""


@dataclass(frozen=True)
class FieldSpec:
    """Per-field invariants.

    A FieldSpec runs only when its key is present in the data dict.
    Presence is enforced separately via :attr:`Schema.REQUIRED_KEYS`.

    Args:
        dtype: Expected ``torch.dtype``.  ``None`` accepts any dtype.
        shape: Expected dimensions.  Each entry may be:
            - ``int``: exact dimension size
            - ``None``: wildcard (any size)
            - ``str``: name of an attribute on the :class:`Schema`
              instance that resolves to an int at validate time.
            Pass ``None`` (the whole field) to skip shape checking.
    """

    dtype: torch.dtype | None = None
    shape: tuple[Any, ...] | None = None

    def resolve_shape(self, schema: "Schema") -> tuple[int | None, ...] | None:
        if self.shape is None:
            return None
        resolved: list[int | None] = []
        for dim in self.shape:
            if isinstance(dim, str):
                # Attribute on the schema instance. A None value means
                # "wildcard for this dim" (e.g. image_height=None).
                val = getattr(schema, dim)
                resolved.append(None if val is None else int(val))
            elif dim is None or isinstance(dim, int):
                resolved.append(dim)
            else:
                raise TypeError(
                    f"FieldSpec.shape entries must be int|None|str; "
                    f"got {type(dim).__name__}: {dim!r}"
                )
        return tuple(resolved)

    def validate(self, value: Any, schema: "Schema", field_name: str) -> None:
        if not torch.is_tensor(value):
            raise SchemaError(
                f"{field_name} must be a torch.Tensor, got "
                f"{type(value).__name__}"
            )
        if self.dtype is not None and value.dtype != self.dtype:
            raise SchemaError(
                f"{field_name} dtype {value.dtype} != expected {self.dtype}"
            )
        expected = self.resolve_shape(schema)
        if expected is not None:
            actual = tuple(value.shape)
            if len(actual) != len(expected):
                raise SchemaError(
                    f"{field_name} has {len(actual)}D shape {actual}, "
                    f"expected {len(expected)}D shape {expected}"
                )
            for i, (a, e) in enumerate(zip(actual, expected)):
                if e is not None and a != e:
                    raise SchemaError(
                        f"{field_name} dim {i}={a}, expected {e} "
                        f"(actual={actual}, expected={expected})"
                    )


class Schema:
    """Base for batch/sample schema contracts.

    Subclasses declare four ClassVar attributes:

      - ``REQUIRED_KEYS``: keys that must be present in every data dict.
      - ``KNOWN_SOURCES``: valid ``source_tag`` values; empty means
        ``source_tag`` is not gate-checked.
      - ``FIELDS``: ``{field_name: FieldSpec}`` mapping.  Field-level
        dtype/shape invariants run for every key that appears in
        ``FIELDS`` *and* in the data.
      - ``_INVARIANTS``: ``{source: callable(data, schema)}`` extra
        checks dispatched on ``source_tag``.

    Subclasses are typically frozen dataclasses so the per-instance
    config (``seq_len``, ``action_dim``, …) can be referenced by
    name from ``FieldSpec.shape`` entries.
    """

    REQUIRED_KEYS: ClassVar[tuple[str, ...]] = ()
    KNOWN_SOURCES: ClassVar[frozenset[str]] = frozenset()
    FIELDS: ClassVar[dict[str, FieldSpec]] = {}
    _INVARIANTS: ClassVar[dict[str, Callable[[Mapping[str, Any], "Schema"], None]]] = {}

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    def validate(
        self, data: Mapping[str, Any], source: str | None = None,
    ) -> None:
        """Validate one sample/batch against the contract.

        Raises :class:`SchemaError` on any violation.  Pass ``source``
        to additionally run the source-specific invariant; omit when
        the schema has no per-source invariants.
        """
        if source is not None and self.KNOWN_SOURCES:
            if source not in self.KNOWN_SOURCES:
                raise SchemaError(
                    f"unknown source {source!r}; expected one of "
                    f"{sorted(self.KNOWN_SOURCES)}"
                )
        missing = [k for k in self.REQUIRED_KEYS if k not in data]
        if missing:
            raise SchemaError(
                f"data missing required keys {missing}; got "
                f"{sorted(data.keys())}"
            )
        if source is not None and "source_tag" in data:
            tag = data["source_tag"]
            if not _source_tag_matches(tag, source):
                raise SchemaError(
                    f"source_tag={tag!r} != expected {source!r} — "
                    f"the adapter chain is likely mis-wired"
                )
        for field_name, field_spec in self.FIELDS.items():
            if field_name in data:
                field_spec.validate(data[field_name], self, field_name)
        if source is not None and source in self._INVARIANTS:
            self._INVARIANTS[source](data, self)

    # --------------------------------------------------------------
    # Probes
    # --------------------------------------------------------------

    def probe_dataset(
        self,
        dataset: Dataset,
        source: str | None = None,
        *,
        n: int = 8,
    ) -> None:
        """Pull up to ``n`` items from ``dataset`` and validate each."""
        if source is not None and self.KNOWN_SOURCES and source not in self.KNOWN_SOURCES:
            raise SchemaError(
                f"unknown source {source!r}; expected one of "
                f"{sorted(self.KNOWN_SOURCES)}"
            )
        length = len(dataset)
        take = min(n, length)
        for i in range(take):
            try:
                sample = dataset[i]
            except Exception as e:
                raise SchemaError(
                    f"dataset[{i}] raised during probe: "
                    f"{type(e).__name__}: {e}"
                ) from e
            try:
                self.validate(sample, source)
            except SchemaError as e:
                raise SchemaError(
                    f"dataset[{i}] (source={source!r}) violates contract: {e}"
                ) from e

    def probe_dataloader(
        self,
        loader: Iterable,
        source: str | None = None,
        *,
        n: int = 1,
    ) -> None:
        """Pull up to ``n`` batches from ``loader`` and validate each.

        This consumes batches from ``loader`` — call it on a fresh
        iterator the consumer is willing to discard, or use
        :meth:`wrap_dataloader` for in-band validation.
        """
        it = iter(loader)
        for i in range(n):
            try:
                batch = next(it)
            except StopIteration:
                return
            except Exception as e:
                raise SchemaError(
                    f"loader yielded error at batch {i}: "
                    f"{type(e).__name__}: {e}"
                ) from e
            try:
                self.validate(batch, source)
            except SchemaError as e:
                raise SchemaError(
                    f"batch {i} (source={source!r}) violates contract: {e}"
                ) from e

    # --------------------------------------------------------------
    # In-band validation wrapper
    # --------------------------------------------------------------

    def wrap_dataloader(
        self,
        loader: Any,
        source: str | None = None,
        *,
        validate_every: int | None = None,
    ) -> "_ValidatingLoader":
        """Wrap a DataLoader so iteration validates batches in-band.

        The first batch is always validated.  Set ``validate_every=N``
        to additionally validate every N-th batch (1-indexed; with
        ``N=10`` you get batches 1, 10, 20, …).  Other batches
        pass through untouched.
        """
        return _ValidatingLoader(loader, self, source, validate_every)


class _ValidatingLoader:
    """Iterator wrapper that runs :meth:`Schema.validate` on batches.

    Forwards every other attribute (``state_dict``, ``load_state_dict``,
    ``__len__``, ``sampler``, etc.) to the wrapped loader, so Lightning
    sees a normal DataLoader.
    """

    def __init__(
        self,
        loader: Any,
        spec: Schema,
        source: str | None,
        validate_every: int | None,
    ):
        self._loader = loader
        self._spec = spec
        self._source = source
        self._validate_every = validate_every

    def __iter__(self) -> Iterator:
        count = 0
        for batch in self._loader:
            count += 1
            should_validate = (
                count == 1
                or (
                    self._validate_every is not None
                    and count % self._validate_every == 0
                )
            )
            if should_validate:
                try:
                    self._spec.validate(batch, self._source)
                except SchemaError as e:
                    raise SchemaError(
                        f"batch {count} (source={self._source!r}) violates "
                        f"contract: {e}"
                    ) from e
            yield batch

    def __len__(self) -> int:
        return len(self._loader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


def _source_tag_matches(tag: Any, source: str) -> bool:
    """Compare a sample's ``source_tag`` against an expected source name.

    Per-sample contracts emit a scalar string tag; collated batches
    surface a list/tuple/0-d-tensor of tags, all of which should
    contain only ``source`` for the batch to be self-consistent.
    """
    if isinstance(tag, str):
        return tag == source
    if isinstance(tag, (list, tuple)):
        return all(t == source for t in tag)
    if torch.is_tensor(tag) and tag.numel() == 1:
        return tag.item() == source
    return False
