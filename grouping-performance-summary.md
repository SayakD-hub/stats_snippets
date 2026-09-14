# Grouping Pipeline Performance Summary

Date: 2026-08-20  
Inspection: `0AXVHCA77S1` (42,566 detection parquet segments, 22 clips)

## Problem

Regular grouping took **~588 s** on a local Prefect run despite 24 CPU cores and 20 worker threads.

## What Did NOT Help

| Change | Effect |
|--------|--------|
| Logging throttle (every 500 segments instead of every file) | ~588s → ~588s — logging was not the bottleneck |
| Segment chunking (split clips into smaller tasks) | ~595s → ~571s — minor gain only |
| Pinning Arrow to 1 thread inside a thread pool | No effect — GIL was the limiter, not Arrow oversubscription |

## Root Cause: GIL + ThreadPoolExecutor

Each segment is cheap (~3 ms read + ~1–2 ms pandas group on ~50k rows, ~1 MB parquet).

With **20 threads**, all workers contend for the Python GIL during pandas work:

| Strategy | Throughput | Extrapolated for 42,566 files |
|----------|------------|-------------------------------|
| 1 thread (serial) | ~195 files/s | ~200 s |
| **20 threads (old)** | **~72 files/s** | **~588 s** ← observed |
| 16 processes (new) | ~920 files/s | ~46 s |

**20 threads were ~2.7× slower than a single thread** due to GIL contention.

Clip layout made load balancing worse (22 clips × ~1,935 segments, 20 workers → two rounds with 18 idle in round 2), but that was secondary to the GIL issue.

## What Actually Sped Things Up

### 1. ProcessPoolExecutor instead of ThreadPoolExecutor

Each worker gets its own Python interpreter, bypassing the GIL.

### 2. `group_segments()` returns DataFrames instead of using a shared Queue

Queues don't work across processes; the parent collects results and feeds the single writer thread.

### 3. `SEGMENT_CHUNK_SIZE = 200`

~213 tasks for 42k files keeps all 20 workers busy instead of 22 clip-sized tasks.

### 4. `_init_grouping_worker()` — `pyarrow.set_cpu_count(1)` per process

Prevents 20 workers × 24 Arrow threads = 480 threads oversubscribing the machine.

## Measured Results

- **End-to-end test** (3,870 segments via `run_grouping_pipeline`): **3.5 s** (~1,121 files/s)
- **Extrapolated full run**: **~38–50 s** for grouping (was ~588 s) — roughly **12–15× faster**
- Low-confidence cascade (~1,854 s with threads) should see a similar factor (~120–200 s)

## Prefect Run Expectations

The timer in `workflows/corrosion_detection/inspection_flows.py` wraps `apply_grouping` and `apply_low_confidence_sampling` directly — same code path as the benchmark. For local runs (no S3):

| Stage | Before | Expected after |
|-------|--------|----------------|
| Regular grouping | ~589 s | ~40–60 s |
| Low-confidence cascade | ~1,854 s | ~120–200 s |

Minor overhead from process pool startup (~1–2 s) and pickling DataFrames back to the parent.

## When Speedup May Be Less

- **S3 / remote runs** — network latency for 42k small reads may become the bottleneck
- **Small inspections** — process spawn overhead is a larger fraction of total time
- **Memory pressure** — 20 processes use more RAM than 20 threads (30 GB should be fine for this workload)

## Unchanged / Correctness Notes

- Per-file grouping logic is unchanged; output groups should be identical (row order in output parquet may differ due to completion order)
- S3 bucket fix (`s3_input_bucket`) is a correctness fix, not a performance fix
- Writer thread remains single-threaded (minor, not the bottleneck)

## Files Changed

- `ilipyutils/src/ilipyutils/grouping.py` — process pool, `group_segments()`, chunking
- `ilipyutils/src/ilipyutils/low_confidence_sampling.py` — same pattern for cascade
- `ilipyutils/tests/test_grouping.py` — test uses `executor_cls=ThreadPoolExecutor` for stubbing
