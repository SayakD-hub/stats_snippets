"""Filter AE dent-detection parquet frames into unique candidate rows.

  1. Load every parquet under ``results_dir``, drop frames at/under the noise
     floor, book raw ``max_prob`` / ``max_prob_track``.
  2. Flag IQR-violating tracks from log win-counts (``max_prob`` above the
     histogram threshold).
  3. For each flagged track, set its extra-keep bar to the chosen percentile
     of *that track's* winning ``max_prob`` values (default 99th).
  4. Keep a frame if ordinary tracks pass 3×0.9 / 2×0.95 / exactly-1×0.99, or
     if no ordinary single fired and exactly one IQR track exceeds *its* bar.
  5. After keep, booking stays raw ``argmax``. ``filter_close_detections``
     collapses nearby ``start`` values (default 0.2 m).

This file does not modify the notebook.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

def parse_prob(prob_str):
    """Convert a stringified probability list to a Python list."""
    if isinstance(prob_str, str):
        try:
            return ast.literal_eval(prob_str)
        except (ValueError, SyntaxError):
            return []
    return prob_str


def _raw_argmax(probs):
    if probs is None or len(probs) == 0:
        return 0.0, -1
    max_i = int(np.argmax(probs))
    return float(probs[max_i]), max_i


def create_df(data_files, frame_range=None, th=0.05):
    """Concatenate per-frame parquet files; drop noise; book raw winners."""
    if frame_range is None:
        data_files_range = list(data_files)
    else:
        prefix = f"frames_{frame_range[0]:012d}_{frame_range[1]:012d}"
        data_files_range = [f for f in data_files if f.name.startswith(prefix)]

    dfs = []
    for path in data_files_range:
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df["prob"] = df["prob"].apply(parse_prob)
        df = df[df["prob"].apply(lambda probs: any(p > th for p in probs))]
        if df.empty:
            continue
        df["prob"] = df["prob"].apply(lambda p: list(p) if p is not None else [])
        winners = df["prob"].apply(_raw_argmax)
        df["max_prob"] = winners.apply(lambda t: t[0])
        df["max_prob_track"] = winners.apply(lambda t: t[1]).astype(int)
        dfs.append(df)

    if not dfs:
        print(f"No valid data found for frame_range={frame_range}")
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0].reset_index(drop=True)


# ---------------------------------------------------------------------------
# IQR outliers + per-track percentile bars
# ---------------------------------------------------------------------------

def outlier_tracks_log_iqr(counts, k=1.5):
    """Flag tracks whose win-count is an upper outlier on the log scale.

    ``counts[i]`` is how often probability index ``i`` was ``max_prob_track``.
    Zero-count bins are ignored for the fence and are never returned.

    ``k`` is Tukey's multiplier, not a percentile. The fence is
    ``Q3 + k * IQR`` on ``log(counts)``. For a normal, ``k=1.5`` sits at
    about +2.70σ (Q3 is +0.6745σ, IQR is ~1.35σ), i.e. the **99.65th
    percentile** one-sided (~0.35% of mass above the fence). Both tails
    together are the usual "~0.7% of a normal sample outside the boxplot
    whiskers." That conversion assumes ``log(counts)`` is normal; it is
    not "the 99.65th percentile of the raw win counts."
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1:
        raise ValueError("counts must be 1-d")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")

    positive = counts > 0
    log_c = np.full(counts.shape, np.nan)
    log_c[positive] = np.log(counts[positive])

    sample = log_c[positive]
    if sample.size < 4:
        return {
            "exclude": [],
            "log_counts": log_c,
            "q1": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
            "upper_fence": np.nan,
        }

    q1, q3 = np.percentile(sample, [25, 75])
    iqr = q3 - q1
    upper = q3 + k * iqr
    exclude = np.flatnonzero(positive & (log_c > upper)).tolist()
    return {
        "exclude": exclude,
        "log_counts": log_c,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "upper_fence": upper,
    }


def win_counts(df, num_tracks, histogram_threshold=0.9):
    """Per-track win counts among frames with ``max_prob`` above the threshold."""
    wins = df.loc[df["max_prob"] > histogram_threshold, "max_prob_track"]
    if wins.empty:
        return np.zeros(num_tracks, dtype=int)
    return np.bincount(wins.astype(int), minlength=num_tracks)


def strict_thresholds_from_win_percentile(df, outlier_tracks, percentile=99):
    """Map each IQR track to the percentile of its own winning ``max_prob``.

    Population matches the notebook plot: rows where that index is
    ``max_prob_track`` on ``all_dfs``, not all scores and not kept rows.
    """
    strict_th = {}
    for t in outlier_tracks:
        maxima = df.loc[df["max_prob_track"] == t, "max_prob"].to_numpy(dtype=float)
        if maxima.size == 0:
            raise ValueError(
                f"IQR track {t} has no winning windows; cannot derive a percentile bar"
            )
        strict_th[int(t)] = float(np.percentile(maxima, percentile))
    return strict_th


# ---------------------------------------------------------------------------
# keep + filter_close_detections
# ---------------------------------------------------------------------------

def keep_row_by_probs(
    probs,
    strict_tracks_set,
    strict_th,
    multi_track_th1=0.9,
    multi_track_th2=0.95,
    single_track_th=0.99,
):
    """Ordinary 3 / 2 / 1 rules, plus a lone IQR hit above that track's bar.

    Strict indices are omitted from every ordinary count. They cannot create
    a 3×0.9, a 2×0.95, or a second 0.99, and they cannot veto an ordinary
    single. Extra keep only if no ordinary single and exactly one strict
    track exceeds *its own* win-percentile bar. There is no n2/n3 inside
    the IQR set.
    """
    if not probs:
        return False

    ordinary = [(i, p) for i, p in enumerate(probs) if i not in strict_tracks_set]
    n3 = sum(1 for _, p in ordinary if p > multi_track_th1)
    n2 = sum(1 for _, p in ordinary if p > multi_track_th2)
    n1_ordinary = sum(1 for _, p in ordinary if p > single_track_th)

    if n3 >= 3 or n2 >= 2 or n1_ordinary == 1:
        return True

    n1_strict = sum(
        1
        for i, p in enumerate(probs)
        if i in strict_tracks_set and p > strict_th[i]
    )
    return n1_ordinary == 0 and n1_strict == 1


def filter_close_detections(df, distance_threshold=0.2):
    """Keep the highest-``max_prob`` row in each 0.2 m ``start`` cluster."""
    if len(df) <= 1:
        return df

    df_sorted = df.sort_values(by="start", ascending=False).reset_index(drop=True)
    starts = df_sorted["start"].values
    probs = df_sorted["max_prob"].values

    groups = []
    current_group = [0]
    for i in range(1, len(df_sorted)):
        if starts[current_group[0]] - starts[i] < distance_threshold:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
    groups.append(current_group)

    kept_indices = [group[int(np.argmax(probs[group]))] for group in groups]
    return df_sorted.iloc[kept_indices].reset_index(drop=True)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def process_detections(
    results_dir,
    num_tracks=22,
    th=0.05,
    histogram_threshold=0.9,
    iqr_k=1.5,
    strict_percentile=99,
    distance_threshold=0.2,
    frame_range=None,
):
    """Full notebook pipeline. Returns ``(all_dfs, filtered_df, final_df, info)``."""
    data_files = sorted(Path(results_dir).rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"no parquet files under {results_dir}")

    all_dfs = create_df(data_files, frame_range=frame_range, th=th)
    if all_dfs.empty:
        raise RuntimeError("create_df returned no rows")

    counts = win_counts(all_dfs, num_tracks, histogram_threshold)
    iqr = outlier_tracks_log_iqr(counts, k=iqr_k)
    outlier_tracks = iqr["exclude"]
    strict_tracks_set = set(outlier_tracks)
    strict_th = strict_thresholds_from_win_percentile(
        all_dfs, outlier_tracks, percentile=strict_percentile
    )

    filtered_df = all_dfs[
        all_dfs["prob"].apply(
            lambda probs: keep_row_by_probs(probs, strict_tracks_set, strict_th)
        )
    ].copy()

    # Booking stays raw argmax of the kept window (not a second IQR gate).
    winners = filtered_df["prob"].apply(_raw_argmax)
    filtered_df["max_prob"] = winners.apply(lambda t: t[0])
    filtered_df["max_prob_track"] = winners.apply(lambda t: t[1]).astype(int)

    final_df = filter_close_detections(filtered_df, distance_threshold=distance_threshold)

    info = {
        "n_files": len(data_files),
        "n_all": len(all_dfs),
        "n_filtered": len(filtered_df),
        "n_final": len(final_df),
        "win_counts": counts.tolist(),
        "iqr_exclude": iqr["exclude"],
        "iqr_log_counts": np.asarray(iqr["log_counts"], dtype=float).tolist(),
        "iqr_q1": iqr["q1"],
        "iqr_q3": iqr["q3"],
        "iqr_upper_fence": iqr["upper_fence"],
        "strict_th": strict_th,
        "upper_fence_count": (
            float(np.exp(iqr["upper_fence"]))
            if np.isfinite(iqr["upper_fence"])
            else float("nan")
        ),
    }
    return all_dfs, filtered_df, final_df, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="/home/sayakdg/stats-experiments/dent_ae_v4.0",
        help="Directory of per-frame AE parquet files",
    )
    parser.add_argument(
        "--inspection-id",
        default="0AXVHCA77S1",
        help="Used only in the default output filename",
    )
    parser.add_argument("--output", default=None, help="Output parquet path")
    parser.add_argument("--num-tracks", type=int, default=22)
    parser.add_argument("--th", type=float, default=0.05, help="Noise floor in create_df")
    parser.add_argument(
        "--histogram-threshold",
        type=float,
        default=0.9,
        help="Win-count floor fed to log-IQR",
    )
    parser.add_argument("--iqr-k", type=float, default=1.5)
    parser.add_argument(
        "--strict-percentile",
        type=float,
        default=99,
        help="Percentile of each IQR track's winning max_prob used as its extra-keep bar",
    )
    parser.add_argument("--distance-threshold", type=float, default=0.2)
    args = parser.parse_args()

    _, _, final_df, info = process_detections(
        results_dir=args.results_dir,
        num_tracks=args.num_tracks,
        th=args.th,
        histogram_threshold=args.histogram_threshold,
        iqr_k=args.iqr_k,
        strict_percentile=args.strict_percentile,
        distance_threshold=args.distance_threshold,
    )

    print(f"files:          {info['n_files']}")
    print(f"all_dfs:        {info['n_all']}")
    print(f"filtered_df:    {info['n_filtered']}")
    print(f"final_df:       {info['n_final']}")
    print(f"IQR exclude:    {info['iqr_exclude']}")
    print(f"count fence:    {info['upper_fence_count']:.1f}")
    for t, bar in info["strict_th"].items():
        print(f"  track {t}: {args.strict_percentile:g}th percentile of wins = {bar:.4f}")

    out = (
        Path(args.output)
        if args.output
        else Path(f"./insp_{args.inspection_id}_final_df_ae.parquet")
    )
    final_df.to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
