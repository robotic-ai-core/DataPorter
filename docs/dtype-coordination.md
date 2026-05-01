# Dtype Coordination (Wire ↔ Working)

## Why this exists

DataPorter has a feature called `dtype_conversions` that downcasts tensors at
collate time:

```yaml
data:
  init_args:
    dtype_conversions:
      - path: "observation.image"
        dtype: "float16"
```

The original intent was bandwidth: pixels travel from worker → main process →
GPU as fp16 instead of fp32, halving DataLoader serialization cost and PCIe
transfer. That part still works.

The latent bug is that a *wire dtype* is not a one-way conversion — it
implicitly fixes the dtype that downstream code (autocast, the model forward,
non-autocast-aware quantized layers) sees. When the wire dtype and the
training-step dtype disagree, you get crashes inside non-autocastable ops.

The motivating production crash:

1. DataPorter downcasts pixels to **fp16** on the wire (saving bandwidth).
2. Lightning enters `precision="bf16-mixed"`, which wraps `training_step` in
   `torch.amp.autocast(dtype=bfloat16)`.
3. Autocast does **not** cross-cast fp16 ↔ bf16 — it leaves fp16 inputs alone
   but casts weights/biases to bf16.
4. `F.conv2d` errors:
   `RuntimeError: Input type (c10::Half) and bias type (c10::BFloat16) should be the same`.

This used to be masked by `IoQuantWrapper` in utorch, which contained an
explicit "if not fp32, upcast to fp32" prelude. utorch removed that
([027a496](https://github.com/.../commit/027a496)) with the architecturally
correct reasoning that **utorch should not own dtype management — Lightning
should**. But Lightning has no idea about DataPorter's wire dtype, so the gap
falls to DataPorter to bridge.

## What dtype coordination is

The fix has two halves:

- The **wire dtype** is the dtype tensors have when they leave the worker
  process. DataPorter sets this to optimize bandwidth.
- The **working dtype** is the dtype tensors have when they enter the model.
  The training framework (Lightning's precision plugin) defines this.

DataPorter owns the bridge. After the device transfer, we upcast wire dtypes
to the working dtype so the model sees something compatible with autocast and
with non-autocastable layers like `Int8Conv2d`.

The contract:

```
worker → collate (wire dtype downcast) → device transfer → upcast to working dtype → model.training_step
                ↑ this is the existing feature                ↑ this is the new piece
```

Both halves are per-key — some tensors stay at full precision throughout
(action vectors, masks, ids), some are downcast for bandwidth and re-upcast
at the device boundary (pixels, large embeddings).

## Why we kept `dtype_conversions` and renamed it

I considered renaming `dtype_conversions` to `precision_coordination` (or
splitting it into `wire_dtype_conversions` + `working_dtype_target`). The
reason I did **not** is:

1. There are configs in the wild (AutoFPV, ProtoWorld, mamba_qat_slm). A hard
   rename would break them.
2. The list-of-dicts schema generalizes naturally: add an optional `working`
   field to each entry, and the new behavior is opt-in per key.
3. The legacy form `[{path, dtype}]` keeps its meaning, but its implicit
   working dtype changes from "whatever the wire dtype is" (the bug) to
   "match the precision plugin" (the fix). This is the right *default* —
   the bug was that there was no working dtype at all.

So `dtype_conversions` stays as the public knob. What changes:

- New optional `working` field per entry. Defaults to `"match"` (= match the
  precision plugin's working dtype).
- The legacy `dtype` field stays as the wire dtype. Old configs become
  equivalent to `{path, dtype, working: "match"}` automatically.
- Users who explicitly want the old wire-only behavior (no upcast) set
  `working: "keep"`.

If you want a structural shift later (e.g., split wire/working at the
config-key level instead of per-entry), the new `working` field gives you a
deprecation path: introduce the new structure, alias the old.

## Configuration

### Default (recommended): wire downcast + match-plugin upcast

```yaml
data:
  init_args:
    dtype_conversions:
      - path: "observation.image"
        dtype: "float16"          # wire dtype (collate)
        # working: "match"        # default — match precision plugin
```

With `precision="bf16-mixed"`, this downcasts to fp16 at collate and upcasts
to bf16 after device transfer. The model sees bf16, autocast is happy.

### Explicit working dtype

```yaml
data:
  init_args:
    dtype_conversions:
      - path: "observation.image"
        dtype: "float16"
        working: "float32"        # always upcast to fp32, ignore plugin
```

Useful when a model has non-autocastable layers that need fp32 specifically
regardless of the precision plugin.

### Disable the upcast (legacy behavior)

```yaml
data:
  init_args:
    dtype_conversions:
      - path: "observation.image"
        dtype: "float16"
        working: "keep"           # do not upcast — wire dtype reaches the model
```

Use this only when you've verified the model handles the wire dtype natively
(no autocast, no quantized layers) — this is the path that produced the
production crash, kept available for backward-compat.

### Per-key heterogeneity

```yaml
data:
  init_args:
    dtype_conversions:
      - path: "observation.image"
        dtype: "float16"
        working: "match"           # downcast + match-plugin upcast
      - path: "action"
        dtype: "float32"
        working: "keep"            # full precision, no upcast (already fp32)
      - path: "metadata.timestamp"
        dtype: "uint16"            # non-floating, working has no effect
```

Non-floating dtypes (`uint8`, `int32`, etc.) are never upcasted — `working`
is silently ignored for them.

## Lightning integration

DataPorter's `BlendedLeRobotDataModule` now overrides Lightning's
`on_after_batch_transfer(batch, dataloader_idx)` hook. After the batch is
moved to GPU, it walks the batch tree applying the working-dtype rules.

For users who don't extend `BlendedLeRobotDataModule`, two opt-ins exist:

1. **`PrecisionCoordinationMixin`** — mix into your `LightningDataModule`:

   ```python
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
   ```

   The mixin provides `on_after_batch_transfer` and `_apply_working_dtype`.

2. **`DtypeCoordinator`** as a standalone helper — call it from your own
   `LightningModule.transfer_batch_to_device` or
   `on_after_batch_transfer`:

   ```python
   from dataporter import DtypeCoordinator

   coord = DtypeCoordinator.from_config(
       dtype_conversions=[{"path": "observation.image", "dtype": "float16"}],
   )

   def on_after_batch_transfer(self, batch, dataloader_idx):
       working_dtype = coord.resolve_working_dtype(self.trainer.precision_plugin)
       return coord.apply_working_dtype(batch, working_dtype)
   ```

Both routes are tested and work with the standalone (no-trainer) case where
`working_dtype = None` makes the upcast a no-op.

## Discovering the working dtype

Source of truth: `trainer.precision_plugin.precision` (public string
attribute, e.g. `"bf16-mixed"`, `"16-mixed"`, `"32-true"`, `"64-true"`).

Mapping:

| `precision` string  | working dtype  | autocast? |
| ------------------- | -------------- | --------- |
| `"bf16-mixed"`      | `bfloat16`     | yes       |
| `"16-mixed"`        | `float16`      | yes       |
| `"bf16-true"`       | `bfloat16`     | no (full) |
| `"16-true"`         | `float16`      | no (full) |
| `"32-true"`         | `float32`      | no (no-op upcast) |
| `"64-true"`         | `float64`      | no        |
| anything else       | error          |           |

The unrecognized-precision case is a hard error rather than a silent fallback,
because it indicates a Lightning version bump that may have added a new
precision mode we need to map deliberately.

When `self.trainer` is `None` (e.g., the datamodule is being used outside
Lightning, or in a `setUp()` test fixture), the working dtype is `None` and
the upcast is a complete no-op — wire dtypes survive to the consumer.

## Failure modes (warn vs error vs silent)

| Situation                                                  | Behavior |
| ---------------------------------------------------------- | -------- |
| `working: "match"`, no Lightning trainer                   | no-op (DEBUG log) |
| `working: "match"`, plugin precision unrecognized          | ERROR (fail fast) |
| Wire `bfloat16` → working `float16`                        | WARN once (lossy round-trip — different exponent ranges) |
| Wire `float16` → working `bfloat16`                        | WARN once (lossy round-trip — fp16 has more mantissa) |
| Wire `float64` → working `float16`/`bfloat16`              | WARN once (large dynamic range loss) |
| Wire `int8`/`uint8` etc. with non-`"keep"` working         | `working` silently ignored on non-float tensors |
| `dtype` and `working` identical at runtime                 | upcast skipped (`tensor.dtype == target` short-circuit) |
| Path declared in `dtype_conversions` not present in batch  | silently no-op (matches existing wire behavior) |
| Path matches but value is not a Tensor                     | silently no-op (matches existing wire behavior) |

The "WARN once" behavior is keyed on `(wire_dtype, working_dtype, path)` so
benign repeated warnings don't flood logs.

## Composition with existing transforms

DataPorter has three transform points already:

1. **`producer_transform`** — runs in the producer pool, on raw frames before
   they hit the shuffle buffer. Output dtype is whatever the transform
   produces (typically `uint8` or `float32`). Not affected.
2. **`AugmentedDataset` train/val transforms** — run per-sample inside the
   dataloader worker. Output dtype is whatever the transform produces.
3. **`KeyBasedDtypeConverter` (wire downcast)** — runs in the collate fn, in
   the worker. Final wire dtype is set here.

The new working-dtype upcast runs in **the main process, after device
transfer**. So the order is:

```
producer_transform → AugmentedDataset → wire downcast (collate, worker) →
    pin_memory → HtoD transfer → working upcast (main process, on-device)
```

This means the upcast runs once per batch on the GPU, on already-pinned
memory. That's cheap (microseconds at typical batch sizes) compared to doing
it on CPU. It also means transforms running before collate see whatever
dtype the dataset produces — they are unaffected by the working dtype.

Note that `tensor.to(dtype)` allocates a new same-shape tensor on the
device when the dtype changes — it is not in-place. At very large batch
sizes that's a real (though small) addition to peak GPU memory. Sized
the same as the input batch's affected tensors.

## Behavior summary for existing configs

If you have an existing config:

```yaml
dtype_conversions:
  - path: "observation.image"
    dtype: "float16"
```

After this change:

- With `precision="bf16-mixed"`: the bug is fixed — pixels arrive at the
  model as bf16 (matches autocast).
- With `precision="16-mixed"`: behavior is unchanged — pixels arrive as fp16
  (matches autocast) — the upcast `fp16 → fp16` is a no-op.
- With `precision="32-true"`: pixels arrive as fp32 (the upcast is no-op
  visually but the conversion is real). This **changes** the prior behavior:
  before, the model would receive fp16 and likely either crash or run a
  silent precision-loss path. After, it receives fp32 — which is what 32-true
  semantically requires. This is a bug fix, not a breaking change.
- Standalone (no Lightning trainer): no upcast happens, prior wire behavior
  preserved.

## What AutoFPV needs to do to opt in

For users of `BlendedLeRobotDataModule` and its subclasses: nothing. The
`on_after_batch_transfer` override is built into the base class.

For users of a custom `L.LightningDataModule` (e.g., the
`tmp/repro_int8_conv2d_bf16_autocast.py` repro):

```python
from dataporter import PrecisionCoordinationMixin
import lightning as L

class MyDM(PrecisionCoordinationMixin, L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self._init_precision_coordination(
            dtype_conversions=[
                {"path": "observation.image", "dtype": "float16"},
            ],
        )
```

The mixin reads `self.trainer.precision_plugin` automatically when
`on_after_batch_transfer` runs, so no extra arguments needed.

## Tests

See `tests/test_dtype_coordination.py` for the matrix:

- bf16-mixed plugin → wire fp16 → working bf16 ✓
- 16-mixed plugin → wire fp16 → working fp16 (no-op upcast) ✓
- 32-true plugin → wire fp16 → working fp32 ✓
- standalone (no trainer) → wire fp16 stays fp16 ✓
- multi-key dtype_conversions with mixed `working` directives ✓
- lossy round-trip warning (fp16 ↔ bf16) ✓
- legacy `dtype_conversions` configs with no `working` field still work ✓
- `working: "keep"` preserves wire dtype ✓
- non-float wire dtype ignores `working` ✓
- `BlendedLeRobotDataModule.on_after_batch_transfer` integration ✓
- The AutoFPV repro at `tmp/repro_int8_conv2d_bf16_autocast.py` runs clean
  after this change.
