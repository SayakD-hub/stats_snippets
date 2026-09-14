# `peaks_grouping_fast_area.py`

Corrosion-probability traces are turned into **one defect per SciPy peak**. Width is not a half-height or a flat threshold run. It is **how far you have to walk from the peak, inside SciPy’s prominence support, to capture a fraction of the excess mass**.

Touching spans stay separate IDs. The left peak wins any overlap.

---

## What it reads / writes

**In (per parquet):** frames on one track, at least `frame_idx` and `id_corro_prob`. Downstream also uses `clip_id`, `track_id`, `odometer_tick`.

**Out:**

| File | Contents |
|---|---|
| `peaks_grouping_area.parquet` | Labeled frames (`defect_marker` > 0), with per-defect mean/count/max of `id_corro_prob` |
| `base_summ_peaks_grouping_area.parquet` | One row per defect with `bin_size >= 0.7` |
| `low_conf_summ_peaks_grouping_area.parquet` | One row per defect with `bin_size < 0.7` |

`vol_conf` is the max `id_corro_prob` on that marker. `frame_length` is inclusive (`max − min + 1`).

---

## Functions

### `defect_center_in_support(y, lo, hi)`

Inside `[lo, hi]`, find the defect center:

- If several samples sit at the max (plateau), use the **midpoint** of that plateau.
- Otherwise use `argmax`.

### `peak_area_bounds_in_support(y, lo, hi, baseline=0, area_frac=0.5)`

1. Center = `defect_center_in_support`.
2. Excess mass: `max(y − baseline, 0)` on the support. Target = `area_frac ×` that sum.
3. Start at the center sample. Repeatedly add the **heavier** of the two neighboring unused samples (tie → **right**) until captured mass ≥ target, or the support is exhausted.

Returns inclusive `(left, right)` still in the same index space as `y`.

### `peak_area_bounds_batch(...)`

Same as above for every peak’s `(left_base, right_base)` from SciPy.

### `label_defects_fast_no_overlap_area(df, x_col, y_col, prominence, area_frac=0.5, wlen=1000, pad=1)`

Full labeling of one series:

1. Zero-pad `y` by `pad` so edge highs can be local maxima.
2. `scipy.signal.find_peaks(y_pad, prominence=prominence, wlen=wlen)`.
3. `peak_prominences(..., wlen=wlen)` → support bases (walk capped at `wlen`).
4. Area bounds inside each support.
5. Unpad (subtract `pad`, clip to `[0, n−1]`).
6. Overlap: if `L_i ≤ R_{i-1}`, set `L_i = R_{i-1} + 1`. Drop empty intervals (`L > R`).
7. Write a **new** `defect_marker` on each remaining `[start, end]`. Adjacent spans do **not** merge.

### `grouper(marked_peaks_df)`

For rows with `defect_marker > 0`, aggregate `id_corro_prob` by marker (`mean`, `count`, `max`) and merge back.

### `process_file(file, ...)`

Load one parquet → label → keep `defect_marker > 0` and `id_corro_prob > 0.05` → `grouper`. Empty → `None`.

### `main()`

1. Hard-coded glob of corrosion v4.2 parquets for inspection `0AXVHCA77S1`.
2. `ProcessPoolExecutor` over files (`tqdm`).
3. Renumber `defect_marker` so IDs are unique across files.
4. Concatenate. Rename `id_corro_prob_max` → `vol_conf`.
5. `bin_size`: left edge of a 0.05-wide half-open bin on `vol_conf` (`[0.10, 0.15, …, 1.00)`, `right=False`).
6. `frame_length` per marker.
7. Drop markers with `vol_conf < 0.05` or `frame_length < 3` (1–2 frame spikes).
8. Write the three parquets. High vs low confidence is **`bin_size >= 0.7`**, not a grouping reject.

---

## Pipeline (one track)

```text
id_corro_prob series
        │
        ▼  pad, find_peaks + peak_prominences (prominence, wlen)
prominence supports
        │
        ▼  grow from plateau/argmax until area_frac of support mass
per-peak [L, R]
        │
        ▼  left peak keeps contested samples; drop empty
defect_marker 1, 2, 3, …
        │
        ▼  length ≥ 3, vol_conf ≥ 0.05
        │
        ▼  bin_size; split 0.7
summaries
```

---

## Parameters (`main`)

| Name | Value | Role |
|---|---|---|
| `prominence` | `0.02` | Peak must stick up this much vs nearby baseline |
| `wlen` | `1000` | Prominence walk window (±500 frames) |
| `area_frac` | `0.6` | Fraction of support mass to label |
| `pad` | `1` | Edge padding |
| overlap | left wins | `L_i = R_{i-1} + 1` |
| min `frame_length` | `3` | Inclusive |
| `bin_size` | 0.10 … 1.00 step 0.05 | Left edge; `vol_conf` in `[edge, edge+0.05)` |
| high / low | `0.7` | On `bin_size` after grouping |

---

## Intent vs threshold-run grouping

Old style: all frames above a cutoff, merge consecutive, maybe split long runs.

This file: only **local maxima**. Width comes from **mass inside the prominence bowl**, not from staying above 0.5. Two peaks that touch get two IDs. `0.7` is a **reporting split**, not the gate that creates groups (`0.05` + length 3 is the gate).
