"""AE keep plus a contiguous-arc union.

The original any-slot rules in ``process_ae_detections`` are unchanged (set A).
This script also keeps windows with a local ordinary-track run (set B):

  * two neighbors whose mean is >= 0.85
  * three neighbors whose mean is >= 0.8

IQR / strict indices break a run and do not count as a neighbor. Index 0
wraps to the last track. A and B are unioned by row (same window once),
then raw ``argmax`` and one ``filter_close_detections``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from process_ae_detections import (
    _raw_argmax,
    create_df,
    filter_close_detections,
    keep_row_by_probs,
    outlier_tracks_log_iqr,
    strict_thresholds_from_win_percentile,
    win_counts,
)


def _ordinary_circular_runs(probs, strict_tracks_set):
    """Consecutive ordinary indices, wrapping 0 to n-1. IQR indices are gaps."""
    n = len(probs)
    if n == 0:
        return []

    runs = []
    i = 0
    while i < n:
        if i in strict_tracks_set:
            i += 1
            continue
        start = i
        while i < n and i not in strict_tracks_set:
            i += 1
        runs.append(list(range(start, i)))

    if (
        len(runs) >= 2
        and 0 not in strict_tracks_set
        and (n - 1) not in strict_tracks_set
        and runs[0][0] == 0
        and runs[-1][-1] == n - 1
    ):
        runs = [runs[-1] + runs[0]] + runs[1:-1]
    return runs


def keep_row_contiguous(
    probs,
    strict_tracks_set,
    pair_mean_th=0.85,
    triple_mean_th=0.8,
):
    """True if any ordinary contiguous pair/triple clears the mean bars."""
    if not probs:
        return False

    for run in _ordinary_circular_runs(probs, strict_tracks_set):
        scores = [float(probs[i]) for i in run]
        for j in range(len(scores) - 1):
            if (scores[j] + scores[j + 1]) / 2.0 >= pair_mean_th:
                return True
        for j in range(len(scores) - 2):
            if (scores[j] + scores[j + 1] + scores[j + 2]) / 3.0 >= triple_mean_th:
                return True
    return False


def process_detections_contig(
    results_dir,
    num_tracks=22,
    th=0.05,
    histogram_threshold=0.9,
    iqr_k=1.5,
    strict_percentile=99,
    distance_threshold=0.2,
    pair_mean_th=0.85,
    triple_mean_th=0.8,
    frame_range=None,
):
    """Original keep ∪ contiguous means. Returns ``(all_dfs, filtered_df, final_df, info)``."""
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

    mask_a = all_dfs["prob"].apply(
        lambda probs: keep_row_by_probs(probs, strict_tracks_set, strict_th)
    )
    mask_b = all_dfs["prob"].apply(
        lambda probs: keep_row_contiguous(
            probs,
            strict_tracks_set,
            pair_mean_th=pair_mean_th,
            triple_mean_th=triple_mean_th,
        )
    )
    filtered_df = all_dfs[mask_a | mask_b].copy()

    winners = filtered_df["prob"].apply(_raw_argmax)
    filtered_df["max_prob"] = winners.apply(lambda t: t[0])
    filtered_df["max_prob_track"] = winners.apply(lambda t: t[1]).astype(int)

    final_df = filter_close_detections(filtered_df, distance_threshold=distance_threshold)

    info = {
        "n_files": len(data_files),
        "n_all": len(all_dfs),
        "n_original": int(mask_a.sum()),
        "n_contig": int(mask_b.sum()),
        "n_both": int((mask_a & mask_b).sum()),
        "n_contig_only": int((~mask_a & mask_b).sum()),
        "n_filtered": len(filtered_df),
        "n_final": len(final_df),
        "win_counts": counts.tolist(),
        "iqr_exclude": iqr["exclude"],
        "iqr_log_counts": np.asarray(iqr["log_counts"], dtype=float).tolist(),
        "iqr_q1": iqr["q1"],
        "iqr_q3": iqr["q3"],
        "iqr_upper_fence": iqr["upper_fence"],
        "strict_th": strict_th,
        "pair_mean_th": pair_mean_th,
        "triple_mean_th": triple_mean_th,
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
    parser.add_argument("--histogram-threshold", type=float, default=0.9)
    parser.add_argument("--iqr-k", type=float, default=1.5)
    parser.add_argument("--strict-percentile", type=float, default=99)
    parser.add_argument("--distance-threshold", type=float, default=0.2)
    parser.add_argument("--pair-mean", type=float, default=0.85)
    parser.add_argument("--triple-mean", type=float, default=0.8)
    args = parser.parse_args()

    _, _, final_df, info = process_detections_contig(
        results_dir=args.results_dir,
        num_tracks=args.num_tracks,
        th=args.th,
        histogram_threshold=args.histogram_threshold,
        iqr_k=args.iqr_k,
        strict_percentile=args.strict_percentile,
        distance_threshold=args.distance_threshold,
        pair_mean_th=args.pair_mean,
        triple_mean_th=args.triple_mean,
    )

    print(f"files:          {info['n_files']}")
    print(f"all_dfs:        {info['n_all']}")
    print(f"original keep:  {info['n_original']}")
    print(f"contiguous:     {info['n_contig']}")
    print(f"both:           {info['n_both']}")
    print(f"contig only:    {info['n_contig_only']}")
    print(f"union:          {info['n_filtered']}")
    print(f"final_df:       {info['n_final']}")
    print(f"IQR exclude:    {info['iqr_exclude']}")
    print(f"count fence:    {info['upper_fence_count']:.1f}")
    for t, bar in info["strict_th"].items():
        print(f"  track {t}: {args.strict_percentile:g}th percentile of wins = {bar:.4f}")

    out = (
        Path(args.output)
        if args.output
        else Path(f"./insp_{args.inspection_id}_final_df_ae_contig.parquet")
    )
    final_df.to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
