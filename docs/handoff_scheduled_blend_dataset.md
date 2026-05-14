# Handoff: ScheduledBlendDataset

**Status:** Spec — not yet implemented
**Owner:** DataPorter agent (incoming)
**Origin:** AutoFPV `mamba_fm` worktree, 2026-05-15
**Parent plan:** Phase **3b** of the DataPorter unified data layer migration (see memory `dataporter_unified_data_layer_plan_2026_05_12.md`). Phase 3 (verbatim port of `BlendedTextDataset`/`WeightedMultiSourceDataset`/the two callbacks) landed on `dev`/`master` at commit `f91aa5d` on 2026-05-15; this work consolidates the four classes onto a single primitive **without breaking the existing API**.

---

## TL;DR

Add a new `ScheduledBlendDataset` + `SourceScheduleCallback` to DataPorter — one N-way weighted-random mixer, one schedule callback. Then **retrofit the four existing classes** (`BlendedTextDataset`, `WeightedMultiSourceDataset`, `MixingScheduleCallback`, `PretrainBlendScheduleCallback`) into thin backward-compatibility wrappers that delegate to the new primitive.

Net effect: one implementation, single source of truth, but every existing YAML config and import path keeps working. Consumers migrate to the new `class_path` in Phase 4 at their own cadence; the wrappers are deleted after that.

Schemas (Phase 1) and text datasets (Phase 2) already landed. This phase is implementable directly on top.

---

## Why

The current text data layer has two parallel weighted mixers nested inside each other:

```
DataLoader → BlendedTextDataset                       (chat vs pretrain coin flip, ratio = MixingScheduleCallback)
                ├── chat_dataset (single, internally pre-blended chat sources)
                └── pretrain_dataset = WeightedMultiSourceDataset     (N-way weighted random, weights = PretrainBlendScheduleCallback)
                        ├── source 0: fineweb-edu
                        ├── source 1: smollm-corpus
                        ├── ...
                        └── source N: v1a
```

The two layers are **conceptually the same operation** (weighted random over sources), separated by a historical accident: `BlendedTextDataset` was the original 2-way mixer; `WeightedMultiSourceDataset` was added later for the N-way pretrain case. This produces:

1. Two scheduling callbacks (`MixingScheduleCallback` for `chat_ratio`, `PretrainBlendScheduleCallback` for the N pretrain weights) with overlapping logic.
2. An artificial chat-vs-pretrain dichotomy that doesn't reflect what users actually want — they want per-source ramps, not a single global chat ratio.
3. Sample-logging callbacks that can't easily report "this sample came from source X" because the source choice happens in `WeightedMultiSourceDataset` and isn't propagated through `BlendedTextDataset`.
4. The val-set construction path in the datamodule has to handle both layers separately.

Unifying lets each chat source ramp independently (current behavior: all chat sources share one `chat_ratio` knob), makes source identification trivial for downstream callbacks, and shrinks the surface area.

---

## Target API

Implement in `dataporter/text/scheduled_blend.py` (or `dataporter/blend/scheduled_blend.py` — wherever the existing DataPorter text package conventions place it):

```python
import torch
from torch.utils.data import Dataset
from typing import Sequence


class ScheduledBlendDataset(Dataset):
    """N-way weighted-random sampling over heterogeneous datasets with
    per-source shared-memory weight tensors.

    Each source has a 1-element shared-memory ``torch.Tensor`` holding the
    (mutable) sampling weight. Weights are normalised at sample time, so
    absolute scale is irrelevant — only ratios matter. A scheduling
    callback can mutate the weight tensors mid-training to implement a
    curriculum (e.g. linear ramps).

    All source datasets must produce ``SampleSpec``-compliant outputs
    (DataPorter's schema discipline). Sources may be heterogeneous: chat
    datasets (on-the-fly template), pretrain Parquet (memory-mapped), or
    any other SampleSpec-compliant Dataset.

    Args:
        sources: list of (dataset, weight_tensor, name) tuples. ``name`` is
            a short string used for sample tagging + logging (e.g.
            ``"v1a"``, ``"fineweb-edu"``, ``"tt"``, ``"no_robots"``).
            Tag is exposed on each sample as ``source_idx`` (int) and is
            available to downstream callbacks via ``self.source_names``.
        virtual_length: if set, ``__len__`` returns this. Otherwise it
            uses the length of the first source (matching the current
            ``WeightedMultiSourceDataset`` convention so the
            DataLoader's RandomSampler has a stable index range).
        zero_weight_fallback: which source index to return when all
            weights are zero (transient state at schedule leading edges).
            Default 0.
    """

    def __init__(
        self,
        sources: Sequence[tuple[Dataset, torch.Tensor, str]],
        virtual_length: int | None = None,
        zero_weight_fallback: int = 0,
    ) -> None: ...

    # --- Public API ---

    @property
    def num_sources(self) -> int: ...

    @property
    def source_names(self) -> list[str]:
        """Indexed list of source names. ``source_names[i]`` is the human
        label of source ``i``."""
        ...

    def set_weight(self, source_idx: int, value: float) -> None:
        """Update one source's weight in shared memory. Visible to in-flight
        ``__getitem__`` calls in worker processes."""
        ...

    def get_weight(self, source_idx: int) -> float: ...

    def get_weights(self) -> list[float]:
        """Snapshot of all current weights, in index order."""
        ...

    # --- Dataset protocol ---

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Pick a source by weighted random, return that source's sample
        with ``source_idx`` tagged. The returned dict is always a fresh
        copy at the top level (no shared-mutation risk for callers that
        add their own keys)."""
        ...
```

### Sample format

The returned dict from `__getitem__` MUST contain at least:

- `input_ids: torch.LongTensor` — sequence of token ids
- `loss_mask: torch.BoolTensor` — same shape as `input_ids`, True where loss should be computed
- `labels: torch.LongTensor` *(optional, sources may add)*
- `source_tag: str` *(existing convention, e.g. `"pretrain_pad"` / `"chat_query"`)*
- **NEW: `source_idx: torch.IntTensor` (scalar, shape `[]`, dtype `int32`)** — the index `0..N-1` of the source this sample came from. Default torch collation produces a `[B]` tensor; this is the desired shape for downstream consumption.

`source_idx` is added by `ScheduledBlendDataset.__getitem__` **after** the inner source's adapter chain has run. If the inner source already happens to set `source_idx`, the outer value should win (idempotent / overwrite). Use `int32` — `default_collate` is unsurprising for `int32`/`int64`; `int16` sometimes upcasts depending on torch version, and the bandwidth difference is in the noise. `TextSampleSpec` invariants for `pretrain_pad` and `chat_query` MUST be updated to accept `source_idx` as a non-required additive key, so wrapper-emitted samples (see below) don't fail the existing schema probe.

### Scheduling callback

Replace `PretrainBlendScheduleCallback` and `MixingScheduleCallback` with a single class:

```python
class SourceScheduleCallback(L.Callback):
    """Mutates ``ScheduledBlendDataset`` per-source weights on a curve.

    Args:
        schedules: list of {source_idx OR source_name, points: [{step, weight}]}
            dicts. ``points`` is the existing N-phase control-point convention
            from ``PretrainBlendScheduleCallback`` (linear interpolation between
            consecutive points; first/last points define the boundary
            behavior).
        log_every_n_steps: cadence for logging current weights to logger.
    """
```

**Key change vs the existing callbacks:** allow `source_name` in addition to `source_idx` for schedule entries. Resolving by name makes configs survive source-reordering and is far less error-prone in YAML. Example:

```yaml
- class_path: dataporter.callbacks.SourceScheduleCallback
  init_args:
    schedules:
    - source_name: v1a
      points:
      - {step: 0.0, weight: 0.0}
      - {step: 0.25, weight: 0.0}
      - {step: 0.75, weight: 0.7}
      - {step: 1.0, weight: 0.7}
    - source_name: nampdn-ai/tiny-textbooks
      points:
      - {step: 0.0, weight: 0.748}
      - {step: 0.75, weight: 0.61}
      - {step: 1.0, weight: 0.61}
    # ... etc
```

Index-based form must remain supported for backward compatibility during the migration.

---

## Behavior contract (must match current implementation)

1. **Weights are normalised at sample time**, not init time. `set_weight` simply writes the shared-memory tensor; the next `__getitem__` reads it.
2. **Shared-memory tensors are visible across worker processes.** Each weight tensor MUST be created and `share_memory_()`-marked in the constructor (before DataLoader workers fork/spawn). Fork-based workers see the same memory page; spawn-based workers receive the shared-memory file-descriptor via pickle. Do not lazily allocate weight tensors — that breaks spawn workers silently.
3. **Edge cases:**
   - All weights zero → return `sources[zero_weight_fallback][idx % len(...)]`. Don't raise.
   - One weight goes negative → clamp to 0 internally (do not raise; schedules may legitimately interpolate through 0).
   - Floating-point edge (`r ≈ total` after the cumulative sum loop) → bucket into the last source. See current `WeightedMultiSourceDataset.__getitem__` for the canonical pattern.
4. **Virtual length pinned to the first source** by default — matches the current convention so `RandomSampler` has a stable index range even when the dominant source rotates.
5. **`idx % len(ds)` per source** — the outer index passes through unchanged, but each inner source wraps its index by its own length. Sources can have different lengths.
6. **Source-name uniqueness** — constructor raises `ValueError` on duplicate names. `set_weight(name=...)` / `get_weight(name=...)` raise `KeyError` on unknown names. Silent no-ops here let YAML typos drift undetected, which is exactly the failure mode this refactor exists to prevent.

---

## What this consolidates (BC wrappers)

Phase 3 already ported the four classes into DataPorter at `src/dataporter/text/blending/`. **None of those files are deleted in this phase.** Instead they get retrofitted into thin backward-compatibility wrappers that delegate to `ScheduledBlendDataset` / `SourceScheduleCallback`:

| File | What changes |
|---|---|
| `src/dataporter/text/blending/weighted_multi_source.py` | `WeightedMultiSourceDataset.__init__` instantiates `ScheduledBlendDataset` with synthetic names `f"source_{i}"`; `__getitem__` delegates. ~20 lines. |
| `src/dataporter/text/blending/blended_dataset.py` | `BlendedTextDataset.__init__` instantiates `ScheduledBlendDataset` with names `"__pretrain__"` / `"__chat__"`; `chat_ratio` setter writes both `w_p ← 1 - r` and `w_c ← r`; `__getitem__` delegates. ~30 lines. |
| `src/dataporter/text/blending/callbacks.py` (`MixingScheduleCallback`) | Continues to write `BlendedTextDataset.chat_ratio` — the wrapper's setter propagates to the underlying `ScheduledBlendDataset` weights. No callback change needed. |
| `src/dataporter/text/blending/callbacks.py` (`PretrainBlendScheduleCallback`) | Continues to walk `dm._pretrain_multi_dataset._weight_tensors[idx]` — the wrapper exposes the new class's per-source tensors at the old attribute name. No callback change needed. |

**Each wrapper emits `DeprecationWarning` on construction** with a one-line pointer to `ScheduledBlendDataset` / `SourceScheduleCallback`. Without this the wrappers will outlive Phase 4 by inertia.

`autofpv/data/blended_text_datamodule.py` (the consuming `LightningDataModule`) is not touched in this phase — it continues to import the wrappers exactly as it does today. The datamodule migration to flat `sources:` config is Phase 4 work owned by the consuming project's agent.

Wrappers and the four old class_path entries get deleted in Phase 5, after Phase 4 consumer migrations land.

---

## Acceptance criteria

1. **API parity tests** — all behaviors from the contract section above have a test:
   - Weight ratios produce expected empirical sampling distribution (chi-square at α=0.01 over ~10k samples).
   - `set_weight` mid-iteration changes the observed distribution.
   - Zero-weight fallback returns from `zero_weight_fallback` index.
   - Negative-weight clamp doesn't raise.
   - Heterogeneous sources (chat + pretrain Parquet mixed in one dataset) work.
   - Constructor raises `ValueError` on duplicate names; `set_weight(name="missing")` raises `KeyError`.

2. **`source_idx` propagation** — every sample has `source_idx` set; downstream `default_collate` produces a `[B]` int32 tensor. `TextSampleSpec.probe_dataset` for `pretrain_pad` / `chat_query` accepts samples carrying the additive `source_idx` key.

3. **Schedule resolution by name** — `SourceScheduleCallback` with `source_name: v1a` finds the right index even if the source list is reordered.

4. **Multi-worker safety** — schedule mutation by the callback (rank 0, main process) becomes visible in worker processes within one step (existing behavior; test that it still holds). Weight tensors are `share_memory_()`-marked in the constructor, not lazily.

5. **Resumability** — when restored from a Lightning checkpoint, the schedule callback resumes the curve from `trainer.global_step` (not step 0). Existing `PretrainBlendScheduleCallback` already handles this; preserve.

6. **YAML config-instantiable** — Lightning CLI can instantiate via `class_path` references.

7. **Wrapper equivalence harness** — for each of the four BC wrappers, a test asserts the externally-observable behavior is preserved:
   - `BlendedTextDataset(pretrain, chat)` with `chat_ratio=r` produces samples whose `source_tag` distribution matches the old implementation at `r ∈ {0.0, 0.3, 0.7, 1.0}` (chi-square at α=0.01, ~10k samples).
   - `WeightedMultiSourceDataset([(ds, w), ...])` with arbitrary weights produces the expected per-source distribution.
   - `MixingScheduleCallback` drives `chat_ratio` along the expected linear curve when invoked at simulated steps (no training loop required).
   - `PretrainBlendScheduleCallback` writes the expected piecewise-linear weight at simulated steps.
   - **Each wrapper emits `DeprecationWarning` on construction** (use `pytest.warns(DeprecationWarning)`).

   Note: this is not "two implementations agreeing" — both code paths route to `ScheduledBlendDataset` underneath, so this tests *the wrapper's translation layer*, not implementation equivalence. That's the correctness-preserving consequence of the retrofit framing.

---

## Out of scope

- Don't migrate `BlendedTextDataModule` itself yet (that's Phase 4 — when consumer projects switch imports). For now, the datamodule continues to import the BC wrappers exactly as it does today.
- Don't change the existing `TextSampleSpec` adapter chain semantics — `ScheduledBlendDataset` consumes already-adapted samples (DropExtras → AddCausalLabels → EnsureLossMask → StampSourceTag). The only `TextSampleSpec` change is accepting `source_idx` as an additive non-required key in the `pretrain_pad` and `chat_query` invariants.
- Don't touch `ChatDataset` or `ChatStreamDataset` — those continue to be one of the sources (just a different kind).
- Don't delete the four BC-wrapper classes in this phase. Deletion is Phase 5, after Phase 4 consumer migrations land.
- The `BlendedSampleLogCallback` (`projects/mamba_qat_slm/mamba_pretrain/callbacks.py:1636`) will need a small upgrade to read `source_idx` and look up `source_names`. Trivial; can land alongside the dataset change or in a follow-up PR.

---

## Resolved decisions (was: open questions)

1. **`source_idx` dtype:** `int32`. Avoids `default_collate` upcast surprises; bandwidth difference is negligible.
2. **Source-name uniqueness:** `ValueError` on duplicates at construction; `KeyError` on unknown name at `set_weight`/`get_weight`. No new exception class — these are config errors, not schema errors.
3. **Schedule clamping at `step > last_point.step`:** hold at the last point's weight. Matches the current `PretrainBlendScheduleCallback` behavior; principle of least surprise.

---

## Reference: current consumer config

The most stressed config is `mamba_fm`'s e75c:

```yaml
# projects/mamba_qat_slm/configs/e75c_22m_v1a_blend_seed42.yaml
data:
  class_path: autofpv.data.blended_text_datamodule.BlendedTextDataModule
  init_args:
    pretrain_sources: [7 sources, see file]
    chat_sources: [3 sources, see file]
    seq_len: 1024
    batch_size: 16
trainer:
  callbacks:
    - class_path: autofpv.data.blended_text_datamodule.PretrainBlendScheduleCallback
      init_args:
        schedules: [7 schedules, 4 of them ramp]
    - class_path: autofpv.data.MixingScheduleCallback
      init_args:
        chat_ratio_end: 0.0  # pure pretrain — chat sources loaded but never sampled
```

This config continues to work unchanged after Phase 3b lands — it routes through the BC wrappers, which delegate to `ScheduledBlendDataset` underneath. Phase 4 (owned by the consuming project) is when the YAML migrates to:

```yaml
data:
  class_path: dataporter.text.BlendedTextDataModule
  init_args:
    sources: [10 sources flat — 7 pretrain + 3 chat — each with name + weight + ramp]
    seq_len: 1024
    batch_size: 16
trainer:
  callbacks:
    - class_path: dataporter.callbacks.SourceScheduleCallback
      init_args:
        schedules: [N schedules keyed by source_name]
```

The wrapper-equivalence harness (AC #7) can run against a stripped-down version of e75c that exercises the same source-mix shape with synthetic data — no need to pull the full corpus into the unit test environment.

---

## Cross-references

- Memory: `dataporter_unified_data_layer_plan_2026_05_12.md` — broader 5-phase migration this is part of
- Memory: `tinytext_synthesis_sketchpad_2026_05_14.md` — v1a corpus consumer, will exercise the new dataset
- Code: `autofpv/data/blended_text_datamodule.py` — what's being replaced
- Code: `autofpv/data/sample_spec.py` — schema discipline (migrating into DataPorter, Phase 1)
- Code: `projects/mamba_qat_slm/mamba_pretrain/callbacks.py:1636` — `BlendedSampleLogCallback`, needs minor upgrade
