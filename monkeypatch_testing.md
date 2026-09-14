# Monkeypatch testing (pytest)

Spy on collaborators. Do not run I/O. Do not stub the math you actually care about.

## What `monkeypatch` does

Pytest’s `monkeypatch` fixture replaces an attribute for one test, then restores it.

```python
monkeypatch.setattr(module, "name", fake)
```

The production function still runs. When it looks up `name` on that module, it gets `fake`.

Patch the **module that uses the name**, not necessarily where it was defined.

| Production code | Patch |
|---|---|
| `from pkg import f` then `f()` | `monkeypatch.setattr(the_file_module, "f", fake)` |
| `import pkg` then `pkg.f()` | `monkeypatch.setattr(pkg, "f", fake)` |

The fake should:

1. Record the arguments you will assert.
2. Return something the caller will accept.
3. Avoid real disk, S3, and subprocesses.

After the test, originals come back even if the assert failed.

---

## Example: `peaks_grouping_fast_area.py`

File: `/home/sayakdg/stats-experiments/peaks_grouping_fast_area.py`

Two layers:

| Layer | Functions | How to test |
|---|---|---|
| Math | `defect_center_in_support`, `peak_area_bounds_in_support`, `label_defects_fast_no_overlap_area`, `grouper` | Real numpy / tiny DataFrames. **No monkeypatch.** |
| Orchestration | `process_file`, `main` | Monkeypatch parquet, glob, pool, writes. |

`label_defects_fast_no_overlap_area` calls SciPy `find_peaks`. That is the algorithm. Feed it a 20-sample series with one obvious bump and assert `defect_marker`. Do not fake `find_peaks` unless you are only checking that `wlen` was forwarded.

Monkeypatch belongs on the **edges**:

```161:184:/home/sayakdg/stats-experiments/peaks_grouping_fast_area.py
def process_file(file: str, ...) -> pd.DataFrame | None:
    df = pd.read_parquet(file)
    marked_peaks_df = label_defects_fast_no_overlap_area(...)
    ...
    marked_peaks_df = grouper(marked_peaks_df)
    return marked_peaks_df
```

```187:242:/home/sayakdg/stats-experiments/peaks_grouping_fast_area.py
def main() -> None:
    files_track = sorted(glob.glob("/home/sayakdg/ILIWorkflows/.../*parquet"))
    with ProcessPoolExecutor() as pool:
        results = list(tqdm(pool.map(worker, files_track), ...))
    ...
    defectDF.to_parquet("peaks_grouping_area.parquet", index=False)
```

That script does `import glob` and `import pandas as pd`. Patch those names **on the script module** (after importing it, e.g. `import peaks_grouping_fast_area as pga`).

### 1. `process_file`: fake the parquet, keep the peak logic

You want to know: given a small frame, does `process_file` return `None` when every row is noise, and a marked table when there is a real peak?

```python
import pandas as pd
import pytest
import peaks_grouping_fast_area as pga


def test_process_file_returns_none_when_no_defects(monkeypatch):
    seen = {}

    def fake_read_parquet(path):
        seen["path"] = path
        return pd.DataFrame(
            {
                "frame_idx": range(8),
                "id_corro_prob": [0.01] * 8,
            }
        )

    monkeypatch.setattr(pga.pd, "read_parquet", fake_read_parquet)

    out = pga.process_file("/tmp/clip.parquet")
    assert seen["path"] == "/tmp/clip.parquet"
    assert out is None
```

`label_defects_fast_no_overlap_area` still runs. You only skipped disk.

To assert grouping on a synthetic peak, put a bump in `id_corro_prob` (e.g. a few samples near 1.0) and assert `defect_marker` / `vol_conf` on the returned frame. Still no patch of SciPy.

### 2. `main`: fake glob, workers, and writes

`main` is a batch job: discover files, map workers, concat, `to_parquet` three times. A unit test should not scan `data-files/` or start a process pool.

**Process pools and monkeypatch:** `ProcessPoolExecutor.map` runs `process_file` in **child processes**. Patches in the parent do not apply there. In tests, replace the pool with in-process `map`, or call `process_file` yourself and do not go through `main`’s executor.

Sketch for `main`:

```python
def test_main_writes_three_parquets_and_does_not_touch_disk(monkeypatch, tmp_path):
    seen = {"glob": None, "writes": []}

    monkeypatch.setattr(
        pga.glob,
        "glob",
        lambda pattern: (seen.update(glob=pattern) or ["a.parquet", "b.parquet"]),
    )

    def fake_process_file(path, **kwargs):
        return pd.DataFrame(
            {
                "defect_marker": [1, 1],
                "frame_idx": [10, 11],
                "id_corro_prob": [0.9, 0.91],
                "id_corro_prob_mean": [0.9, 0.9],
                "id_corro_prob_max": [0.91, 0.91],
                "clip_id": ["c0", "c0"],
                "track_id": [0, 0],
                "odometer_tick": [100, 101],
            }
        )

    monkeypatch.setattr(pga, "process_file", fake_process_file)

    class ImmediatePool:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, fn, files):
            return [fn(f) for f in files]

    monkeypatch.setattr(pga, "ProcessPoolExecutor", lambda: ImmediatePool())
    monkeypatch.setattr(pga.tqdm, "__call__", lambda it, **k: it)  # optional

    def fake_to_parquet(self, path, **kwargs):
        seen["writes"].append(path)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    pga.main()

    assert "corrosion_v4.2" in seen["glob"]
    assert seen["writes"] == [
        "peaks_grouping_area.parquet",
        "base_summ_peaks_grouping_area.parquet",
        "low_conf_summ_peaks_grouping_area.parquet",
    ]
```

That pins **wiring**: glob pattern, three output names, concat of worker results. It does not prove area-growth is correct. That belongs in tests of `peak_area_bounds_in_support` with a hand-built `y`.

### 3. What not to patch

| Call | Why not |
|---|---|
| `signal.find_peaks` / `peak_prominences` | That *is* the detector. |
| `peak_area_bounds_in_support` from `label_defects_*` | You would only test overlap clipping, not area. |
| `grouper` internals | A 5-row DataFrame with `defect_marker` is enough. |

If a test fakes `find_peaks` to return `[3]` and then asserts markers around 3, you have only tested indexing and `pad`, not prominence.

---

## Call graph (orchestration test)

```
test
  monkeypatch pga.pd.read_parquet          → records path, returns tiny DataFrame
  pga.process_file("/tmp/clip.parquet")
      pd.read_parquet(...)                 # fake
      label_defects_fast_no_overlap_area   # real
      grouper                              # real
  assert return value / recorded path
  pytest restores pd.read_parquet
```

## Same idea in a workflow wrapper

OD `grouping.apply_grouping` is the same pattern: patch `match_clip_data_files` and `run_grouping_pipeline` on the **grouping module** (the `from ilipyutils... import` names), record `s3_bucket` / `s3_output_bucket` / key, never hit S3.

Rule: patch I/O and other services. Run the function under test. Assert the arguments you forwarded and the small result you care about.
