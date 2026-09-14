# Peak-area grouping for ID / OD corrosion

**Status:** proposed  
**Scope:** `ILIWorkflows` — `ilipyutils.grouping`, `corrosion_detection`, `od_corrosion_detection`  
**Reference implementation:** `stats2/stats-experiments/peaks_grouping_fast_area.py`  
**Not in scope:** crack grouping, dent AE windows, calibration model training (consumes the new tables)

## 1. Why

Regular grouping is run-length on `prob >= grouping_threshold`, then a max-span split and a `filter_threshold` on `max(prob)`. Low-confidence sampling **re-groups** the same files at every score bin, cascades nested intervals, and subsamples. That second pass is expensive and does not produce a defect-level score: the same geometry can exist at several thresholds and is kept once at the highest cut.

Peak-area grouping finds defects once on the **full** probability series. Each defect is one grouped row with `vol_conf = max(prob)` on its support. The same binning is applied to every defect. Then the table is written as **two grouped parquets**:

| File | Rule |
| --- | --- |
| Regular grouped | `vol_conf >= 0.7` |
| Low-prob sample | `0.05 <= vol_conf < 0.7` |

Peaks with `vol_conf < 0.05` are discarded as noise (frozen floor; see §5).

Both files have the same schema (defect summary, including `bin_size`). There is no second grouping algorithm, no cascade, and no subsample pipeline. Inspection wall-clock becomes **one grouping stage instead of two**. The science is a defect-level score you can put on a reliability diagram: P(true corrosion | model max in bin \(t\)).

## 2. Current production

| Piece | Location | Behavior |
| --- | --- | --- |
| Shared grouping | `ilipyutils/src/ilipyutils/grouping.py` | `transform_detection_to_groups_for_task` → `group_anomalies_from_detections` (per 50k parquet) → `run_grouping_pipeline` |
| ID / OD callers | `workflows/{corrosion,od_corrosion}_detection/grouping.py` | Same helper; `TaskGroupingConfig` picks `id_corro_prob` / `od_corro_det_prob` |
| Parameters | `GroupingParameters` | `min_continuous_frames`, `max_continuous_frames`, `grouping_threshold`, `filter_threshold` |
| Grouped filename | `generate_s3_filename()` | `min5_max2000_group0.5_filter0.7.parquet` |
| Low-conf | `ilipyutils/low_confidence_sampling.py` | Multi-threshold cascade + `sample_sizes`; wired on ID per model and OD `inspection_flow` |

`group_anomalies_from_detections` already processes **one shard parquet at a time** (same granularity as the experiment script’s `process_file`). Prefect / the existing thread pool parallelize by `(clip_id, profile)`. No `ProcessPoolExecutor`.

Downstream **corrosion_sizing** reads a full S3 key and expects `clip_id`, `start_odo_ticks`, `end_odo_ticks`, `start_frame_idx`, `end_frame_idx`, `vol_conf` (and `anomaly_type`). It does not derive the grouped filename.

ID `ribbon_inspection_flow` runs regular grouping only (no low-conf today). After this change it uses peak grouping like clip inspection and still writes both parquets.

`DentModel` / `LaminationModel` / `SeamWeldModel` share `transform_detection_to_groups_for_task`. Replacing it in place changes those tasks too unless gated (see §8).

## 3. Algorithm (from the experiment script)

Per shard dataframe, sort by `frame_idx`. Probability column = `task_config.prob_column`. **No** score prefilter. The detection parquet is already one row per frame.

Frozen constants (not job parameters; canonical list in §5): `PROMINENCE=0.02`, `AREA_FRAC=0.8`, `PAD=1`, `VOL_CONF_SPLIT=0.7`, `VOL_CONF_FLOOR=0.05`.

Steps:

1. `y = prob.to_numpy()`; `y_pad = pad(y, pad, constant=0)`.
2. `scipy.signal.find_peaks(y_pad, prominence=0.02)`.
3. For each peak: valley walk to baseline 0 (`peak_valley_bounds`); defect center = plateau midpoint or argmax; grow left/right until `area_frac=0.8` of excess mass in the valley (`peak_area_bounds`). Unpad / clip indices to `[0, n)`.
4. Overlaps: contested span goes to the **left** peak (`L_i = R_{i-1} + 1`). Invalid (`L > R`) peaks are dropped.
5. Aggregate **directly to defect summary** (no per-frame outer merge — that was for plots):

   | Column | How |
   | --- | --- |
   | `clip_id`, `track_id` | first |
   | `start_frame_idx`, `end_frame_idx` | min / max of support |
   | `start_odo_ticks`, `end_odo_ticks` | first/last on sorted frames in support (match current grouping) |
   | `vol_conf` | max(prob) |
   | `anomaly_type` | `task_config.anomaly_type` |
   | `bin_size` | left edge of 0.05-wide bins on `vol_conf`. Applied to defects that survive length and the 0.05 floor, **before** the 0.7 file split. |

6. Drop defects with inclusive `frame_length = end_frame_idx - start_frame_idx + 1` **below** `min_frame_length` (Prefect-exposed; default **5**). Drop defects with `vol_conf < VOL_CONF_FLOOR` (`0.05`) as noise. Remaining rows go to one of the two files.

`pad` only lets `find_peaks` return endpoints of the shard. It does **not** stitch across 50k files. A defect straddling `frames_050000_100000` remains two defects. Same as the script; accepted.

**Identity:** do not write `defect_marker`. It is a per-shard counter and collides after concat. Natural key is `(clip_id, start_frame_idx)` (plus `track_id` if desired). Optional integer `group_id` only if assigned in the inspection-level writer after concat.

## 4. Two outputs, one pass

After aggregation, length filter, and the 0.05 floor, split on `vol_conf` (fixed 0.7):

| Table | Rule | Role |
| --- | --- | --- |
| Regular grouped parquet | `vol_conf >= 0.7` | Sizing, bookmarks, current consumers |
| Low-prob-sample parquet | `0.05 <= vol_conf < 0.7` | Review / calibration sample |

Same grouped schema and same `bin_size` rule in both. This is **not** a subsample and **not** cascade. Every peak that survives length and the floor appears in exactly one file. Rows with `vol_conf < 0.05` are not written.

`run_grouping_pipeline` today writes one key. Extend it (or a thin wrapper) to write two keys from the same in-memory groups, still using `s3_bucket` / `s3_output_bucket` as now (reads on inference bucket, writes on grouping bucket).

### Proposed keys

Keep the `grouping/` directory. **Do not** encode frozen constants in the key. Keep the **ML model head** (`output_prefix` / `version_segment`) on the object name as today. The only extra designator is `low_prob_sample` for `0.05 <= vol_conf < 0.7`. The only variable token is `min_frame_length`.

ID:

```
.../corrosion_detection/{output_prefix}/grouping/{output_prefix}_grouped_min{min_frame_length}.parquet
.../corrosion_detection/{output_prefix}/grouping/{output_prefix}_grouped_low_prob_sample_min{min_frame_length}.parquet
```

Example: `corrosion_v4.2_grouped_min5.parquet`, `corrosion_v4.2_grouped_low_prob_sample_min5.parquet`.

OD (head on the object as well as in the path, so both workflows match):

```
.../od_corrosion_detection/{version_segment}/grouping/{version_segment}_grouped_min{min_frame_length}.parquet
.../od_corrosion_detection/{version_segment}/grouping/{version_segment}_grouped_low_prob_sample_min{min_frame_length}.parquet
```

Example: `od_corrosion_v4.0_grouped_min5.parquet`, `od_corrosion_v4.0_grouped_low_prob_sample_min5.parquet`.

Old run-length names (`min5_max2000_group0.5_filter0.7.parquet`) are not reused, so existing objects stay put.

## 5. Parameters

### Frozen parameters (explicit)

These are **module constants**. They are not `GroupingParameters` fields, not Prefect job inputs, and **not** encoded in S3 keys. Changing any of them is a code change (and a Changie), not a deployment form tweak.

| Name | Value | Role |
| --- | --- | --- |
| `PROMINENCE` | `0.02` | `scipy.signal.find_peaks` |
| `AREA_FRAC` | `0.8` | fraction of valley excess mass used as defect width |
| `PAD` | `1` | zero-pad so endpoint frames can be peaks |
| `VOL_CONF_SPLIT` | `0.7` | regular file vs `low_prob_sample` file |
| `VOL_CONF_FLOOR` | `0.05` | drop peaks below this as noise; they appear in **neither** parquet |

`VOL_CONF_FLOOR` is stricter than the experiment script’s 0.1 cut. Below 0.05 is treated as noise, not as a calibration bin.

### Exposed (Prefect + filename)

| Field | Default | Notes |
| --- | --- | --- |
| `min_frame_length` | `5` | Inclusive frame span; drop defects shorter than this. Replaces `min_continuous_frames`. |

`GroupingParameters` is replaced in place with this single field (`ge=1`). Drop `grouping_threshold`, `filter_threshold`, `max_continuous_frames`. Do **not** add `prominence` or `area_frac` fields.

`generate_s3_filename()` for the regular file: `min{min_frame_length}.parquet`. Low-prob sibling: `low_prob_sample_min{min_frame_length}.parquet`. Callers prefix the model head as in §4.

Remove entirely: `LowConfidenceSamplingParameters` and the low-conf pipeline.

## 6. What to delete

- `ilipyutils/src/ilipyutils/low_confidence_sampling.py`
- `ilipyutils/tests/test_low_confidence_sampling.py`
- `LowConfidenceSamplingParameters` in `config.py`
- `apply_low_confidence_sampling` and `_low_conf_output_filename` (ID and OD)
- `inspection_flow` low-conf branches and `--low-confidence-sampling`
- `MLModel.low_confidence_sampling` on ID models
- README / Changie / tests that describe cascade bins and `sub_sampled_*`

Keep `match_clip_data_files`, queue writer, provenance stamp, cross-bucket I/O.

## 7. Implementation plan

1. Port `peak_valley_bounds`, `defect_center`, `peak_area_bounds`, `peak_area_bounds_batch`, `label_defects_fast_no_overlap_area` into `ilipyutils.grouping` (or `ilipyutils.peak_grouping`) with the frozen constants. Unit tests from synthetic series: isolated peak, plateau, left-wins overlap, empty, pad endpoints. Do **not** port `grouper()` (outer merge).
2. Rewrite `transform_detection_to_groups_for_task` to call that path and return the defect-summary schema (plus `bin_size`). Preserve `TaskGroupingConfig`. Apply `min_frame_length` and `VOL_CONF_FLOOR` here.
3. Extend `run_grouping_pipeline` to split at `VOL_CONF_SPLIT` (0.7) and write two parquets. Empty typed writes if no peaks in that half.
4. Shrink `GroupingParameters` to `min_frame_length`; update filename helpers; `apply_grouping` only (no second apply).
5. Strip low-conf from flows, inference models, READMEs, tests.
6. Point `corrosion_sizing` local examples at `{head}_grouped_min5.parquet` (jobs already pass full keys).
7. Changie: grouping algorithm change + removal of low-conf sampling (Changed / Removed as appropriate).

`scipy` is required in `ilipyutils` / workflow images (`scipy.signal.find_peaks`). Confirm it is already a dependency; add if not.

## 8. Shared helper / other tasks

Replacing `transform_detection_to_groups_for_task` in place applies peak grouping to **every** `TaskGroupingConfig` (corrosion, OD, dent, lamination, seamweld). That is intended if those series are the same kind of per-frame probability. If dent/lam should stay run-length, add `grouping_method: peak_area | run_length` (default `peak_area` for corrosion/OD only). **Decision needed before merge.** Recommendation: peak_area for all tasks that already use this function, unless a dent/lam owner objects — one instance definition in one place.

## 9. Downstream and jobs

- **Sizing:** pass the new regular grouped key (`…_grouped_min5.parquet`). Schema compatibility: keep existing columns; extra columns (`bin_size`) are ignored if unread.
- **Prefect:** old `GroupingParameters` JSON (`grouping_threshold`, …) will fail `extra=forbid` after the field swap. Jobs only need `min_frame_length` (or omit for 5). Document a one-time migration. Old grouped objects remain on S3 under old names.
- **Existing low-conf objects** (`*_low_conf_bins_*`, `sub_sampled_*`) are frozen artifacts; nothing regenerates them.

## 10. Tests

- Peak geometry (synthetic `y`) with frozen prominence / area_frac.
- `transform_detection_to_groups_for_task` vs a small fixture from the experiment (mind `area_frac` 0.8 vs script 0.85 if comparing numerically).
- Split: a `vol_conf=0.85` defect only in regular; `0.4` only in `low_prob_sample`; `0.04` in neither.
- Length: a 3-frame peak is dropped when `min_frame_length=5`.
- Pipeline: two output paths / S3 keys; empty input → two empty typed parquets.
- ID/OD `test_output_paths.py` for new filenames; remove low-conf path tests.
- Existing grouping tests that encode run-length / `array_split` / min-max frames **rewrite** for peaks.

## 11. Performance

The speed win is **removing the entire low-conf pass** (repeat grouping × bins + cascade), not beating filtered run-length on a quiet shard. Peak CPU is `find_peaks` on ~50k samples plus a Python walk per peak; experiment timing was acceptable. Inspection remains I/O-bound. No multiprocess in grouping; keep the existing clip/profile thread pool.

## 12. Calibration (follow-on, not this PR)

`bin_size` on both parquets is the X of a reliability diagram. Fit isotonic / logistic against GT or review labels (low-prob file plus a sample of the regular file). Apply the map to `vol_conf` at serving time. This PR only **produces** the defect-level score and the bucket column.

## 13. Risks

| Risk | Mitigation |
| --- | --- |
| Instance counts ≠ run-length groups | Expected; do not mix old and new keys in one sizing job |
| Shard-split defects | Documented; same as experiment |
| Frozen prominence 0.02 too many peaks on noisy clips | Length filter + 0.05 floor; changing frozen constants is a code change, not a job param |
| `area_frac` 0.8 vs experiment 0.85 | Document; re-check extents if comparing to old experiment parquets |
| Floor 0.05 vs experiment 0.1 | Low file will include 0.05–0.1; expected |
| Prefect param break | Changelog + example job JSON (`min_frame_length` only) |

## 14. Done when

- ID and OD inspection grouping uses peaks only; low-conf module gone.
- Two grouped parquets per model head as in §4.
- Prominence, area_frac, pad, 0.7 split, and 0.05 floor are frozen constants (not Prefect fields).
- Unit tests green; one local or Prefect run on a known inspection is sensible vs the experiment (allowing for `area_frac` 0.8).
- Sizing example key updated; README describes frozen peak constants, `min_frame_length`, and the two filenames.
