import glob
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import scipy.signal as signal
from tqdm import tqdm


def defect_center_in_support(y: np.ndarray, lo: int, hi: int) -> int:
    """Return plateau midpoint or argmax within SciPy prominence support."""
    seg = y[lo : hi + 1]
    ymax = seg.max()
    tol = max(1e-9, 1e-6 * ymax)
    at_max = np.flatnonzero(np.abs(seg - ymax) <= tol)
    if len(at_max) > 1:
        return lo + (at_max[0] + at_max[-1]) // 2
    return lo + int(np.argmax(seg))


def peak_area_bounds_in_support(
    y: np.ndarray,
    lo: int,
    hi: int,
    baseline: float = 0.0,
    area_frac: float = 0.5,
) -> tuple[int, int]:
    """
    Anchor at defect center (plateau midpoint or true max), then grow outward
    asymmetrically until `area_frac` of excess mass within SciPy prominence
    bases is captured.
    """
    if not 0.0 < area_frac <= 1.0:
        raise ValueError("area_frac must be in (0.0, 1.0]")

    lo = max(int(lo), 0)
    hi = min(int(hi), len(y) - 1)
    if lo > hi:
        return lo, hi

    ref = defect_center_in_support(y, lo, hi)
    seg = np.maximum(y[lo : hi + 1] - baseline, 0.0)
    total = seg.sum()
    if total == 0:
        return ref, ref

    target = area_frac * total
    r = int(np.clip(ref - lo, 0, len(seg) - 1))
    left = right = r
    captured = seg[r]

    while captured < target and (left > 0 or right < len(seg) - 1):
        mass_left = seg[left - 1] if left > 0 else -1.0
        mass_right = seg[right + 1] if right < len(seg) - 1 else -1.0
        if mass_right >= mass_left:
            right += 1
            captured += seg[right]
        else:
            left -= 1
            captured += seg[left]

    return lo + left, lo + right


def peak_area_bounds_batch(
    y: np.ndarray,
    peaks: np.ndarray,
    left_bases: np.ndarray,
    right_bases: np.ndarray,
    baseline: float = 0.0,
    area_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    left_idx = np.empty(len(peaks), dtype=int)
    right_idx = np.empty(len(peaks), dtype=int)
    for i in range(len(peaks)):
        left_idx[i], right_idx[i] = peak_area_bounds_in_support(
            y,
            int(left_bases[i]),
            int(right_bases[i]),
            baseline,
            area_frac,
        )
    return left_idx, right_idx


def label_defects_fast_no_overlap_area(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    prominence: float,
    area_frac: float = 0.5,
    wlen: int = 1000,
    pad: int = 1,
) -> pd.DataFrame:
    """
    Find peaks by prominence, take SciPy prominence bases as support (capped
    by `wlen`), anchor each defect at plateau center / true max, assign width
    via greedy asymmetric area growth within that support, resolve overlaps by
    giving contested span to the left peak (L_i = R_{i-1} + 1). Each remaining
    peak keeps its own defect_marker (adjacent spans do not merge).
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"Missing column(s): {x_col}, {y_col}")

    out = df.copy()
    y = out[y_col].to_numpy(dtype=float)
    n = len(out)

    y_pad = np.pad(y, pad_width=pad, mode="constant", constant_values=0.0)
    peaks, _ = signal.find_peaks(y_pad, prominence=prominence, wlen=wlen)
    out["defect_marker"] = 0
    if len(peaks) == 0:
        return out

    _, left_bases, right_bases = signal.peak_prominences(y_pad, peaks, wlen=wlen)
    left_idx, right_idx = peak_area_bounds_batch(
        y_pad,
        peaks,
        left_bases,
        right_bases,
        baseline=0.0,
        area_frac=area_frac,
    )
    left_idx = np.clip(left_idx - pad, 0, n - 1)
    right_idx = np.clip(right_idx - pad, 0, n - 1)

    for i in range(1, len(peaks)):
        if left_idx[i] <= right_idx[i - 1]:
            left_idx[i] = right_idx[i - 1] + 1

    valid = left_idx <= right_idx
    left_idx = left_idx[valid]
    right_idx = right_idx[valid]

    marker = np.zeros(n, dtype=np.int32)
    defect_id = 0
    for start, end in zip(left_idx, right_idx):
        defect_id += 1
        marker[start : end + 1] = defect_id

    out["defect_marker"] = marker
    return out


def grouper(marked_peaks_df: pd.DataFrame) -> pd.DataFrame:
    marked_peaks_grp_df = marked_peaks_df.query("defect_marker > 0")[
        ["id_corro_prob", "defect_marker"]
    ].groupby("defect_marker").agg(["mean", "count", "max"]).reset_index()
    marked_peaks_grp_df.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in marked_peaks_grp_df.columns
    ]
    return marked_peaks_df.merge(
        marked_peaks_grp_df,
        on="defect_marker",
        how="outer",
    ).fillna(0)


def process_file(
    file: str,
    prominence_thresh: float = 0.02,
    area_frac: float = 0.5,
    wlen: int = 1000,
) -> pd.DataFrame | None:
    df = pd.read_parquet(file)
    marked_peaks_df = label_defects_fast_no_overlap_area(
        df,
        "frame_idx",
        "id_corro_prob",
        prominence_thresh,
        area_frac,
        wlen=wlen,
    )
    marked_peaks_df = marked_peaks_df.query(
        "defect_marker > 0 and id_corro_prob > 0.05"
    )
    if len(marked_peaks_df) == 0:
        return None
    marked_peaks_df = grouper(marked_peaks_df)
    if len(marked_peaks_df) == 0:
        return None
    return marked_peaks_df


def main() -> None:
    prominence_thresh = 0.02
    area_frac = 0.6
    wlen = 1000

    files_track = sorted(
        glob.glob(
            "/home/sayakdg/ILIWorkflows/workflows/corrosion_detection/"
            "data-files/0AXVHCA77S1/corrosion_detection/corrosion_v4.2/*/*/0/*parquet"
        )
    )

    worker = partial(
        process_file,
        prominence_thresh=prominence_thresh,
        area_frac=area_frac,
        wlen=wlen,
    )

    parts = []
    defect_id_offset = 0

    with ProcessPoolExecutor() as pool:
        results = list(tqdm(pool.map(worker, files_track), total=len(files_track)))

    for marked_peaks_df in results:
        if marked_peaks_df is None:
            continue
        marked_peaks_df = marked_peaks_df.copy()
        marked_peaks_df["defect_marker"] += defect_id_offset
        defect_id_offset = int(marked_peaks_df["defect_marker"].max())
        parts.append(marked_peaks_df)

    defectDF = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    defectDF = defectDF.rename(columns={"id_corro_prob_max": "vol_conf"})

    edges = np.arange(0.1, 1.0 + 1e-9, 0.05)
    defectDF["bin_size"] = pd.cut(
        defectDF["vol_conf"],
        bins=edges,
        right=False,
        labels=edges[:-1],
    ).astype(float)

    g_frame = defectDF.groupby("defect_marker")["frame_idx"]
    defectDF["frame_length"] = (
        g_frame.transform("max") - g_frame.transform("min") + 1
    )

    # Drop 1–2 frame spikes only; keep anything ≥ 3 frames
    low_prob_markers = defectDF.query("vol_conf < 0.05")["defect_marker"].unique()
    defectDF = defectDF.query(
        "defect_marker not in @low_prob_markers and frame_length >= 3"
    ).reset_index(drop=True)

    defectDF.to_parquet("peaks_grouping_area.parquet", index=False)

    defect_summary = (
        defectDF.groupby("defect_marker", as_index=False)
        .agg(
            clip_id=("clip_id", "first"),
            track_id=("track_id", "first"),
            start_frame_idx=("frame_idx", "min"),
            end_frame_idx=("frame_idx", "max"),
            start_odo_ticks=("odometer_tick", "min"),
            end_odo_ticks=("odometer_tick", "max"),
            vol_conf=("vol_conf", "first"),
            id_corro_prob_mean=("id_corro_prob_mean", "first"),
            bin_size=("bin_size", "first"),
            frame_length=("frame_length", "first"),
        )
        .assign(length_odo=lambda x: x["end_odo_ticks"] - x["start_odo_ticks"])
    )

    defect_summary = defect_summary[
        [
            "defect_marker",
            "clip_id",
            "track_id",
            "start_odo_ticks",
            "end_odo_ticks",
            "start_frame_idx",
            "end_frame_idx",
            "vol_conf",
            "id_corro_prob_mean",
            "bin_size",
            "frame_length",
            "length_odo",
        ]
    ]
    base_defect_summary = defect_summary.query("bin_size >= 0.7").reset_index(drop=True)
    lc_defect_summary = defect_summary.query("bin_size < 0.7").reset_index(drop=True)

    base_defect_summary.to_parquet("base_summ_peaks_grouping_area.parquet", index=False)
    lc_defect_summary.to_parquet("low_conf_summ_peaks_grouping_area.parquet", index=False)


if __name__ == "__main__":
    main()
