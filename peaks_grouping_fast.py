import glob
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy.signal as signal
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------------------- functions ---------------------------------------------------------------------
def label_defects_fast_no_overlap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    prominence: float,
    cutoff: float,
) -> pd.DataFrame:
    """
    Find peaks, take width at `cutoff` (rel_height), resolve overlaps by
    giving the contested span to the left peak (L_i = R_{i-1} + 1), then
    assign defect IDs (new ID only when there is a gap).
    """
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("cutoff must be in [0.0, 1.0]")
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"Missing column(s): {x_col}, {y_col}")

    out = df.copy()
    y = out[y_col].to_numpy(dtype=float)
    n = len(out)

    peaks, _ = signal.find_peaks(y, prominence=prominence)
    out["defect_marker"] = 0
    if len(peaks) == 0:
        return out

    _, _, left_ips, right_ips = signal.peak_widths(y, peaks, rel_height=cutoff)
    left_idx = np.clip(np.round(left_ips).astype(int), 0, n - 1)
    right_idx = np.clip(np.round(right_ips).astype(int), 0, n - 1)

    # Contested area goes to the left peak
    for i in range(1, len(peaks)):
        if left_idx[i] <= right_idx[i - 1]:
            left_idx[i] = right_idx[i - 1] + 1

    # Drop peaks whose interval vanished after the clip
    valid = left_idx <= right_idx
    left_idx = left_idx[valid]
    right_idx = right_idx[valid]

    marker = np.zeros(n, dtype=np.int32)
    defect_id = 0
    prev_end = -2

    for start, end in zip(left_idx, right_idx):
        if start > prev_end + 1:
            defect_id += 1
        marker[start : end + 1] = defect_id
        prev_end = end
    out["defect_marker"] = marker

    return out


def grouper(marked_peaks_df):
    # group by marker
    marked_peaks_grp_df = marked_peaks_df.query("defect_marker > 0")[
        ["id_corro_prob", "defect_marker"]
    ].groupby("defect_marker").agg(["mean", "count", "max"]).reset_index()
    # Flatten the nested columns instantly
    marked_peaks_grp_df.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in marked_peaks_grp_df.columns
    ]
    marked_peaks_df = marked_peaks_df.merge(
        marked_peaks_grp_df,
        left_on="defect_marker",
        right_on="defect_marker",
        how="outer",
    ).fillna(0)
    return marked_peaks_df


def nearest_marker(x, markers):
    x = np.asarray(x, dtype=float)
    idx = np.abs(x[:, None] - markers[None, :]).argmin(axis=1)
    out = markers[idx]
    out[np.isnan(x)] = np.nan
    return out


def process_file(file: str) -> pd.DataFrame | None:
    df = pd.read_parquet(file)
    prominence_thresh = 0.1
    cutoff_threshold = 0.5  # FWHM

    marked_peaks_df = label_defects_fast_no_overlap(
        df, "frame_idx", "id_corro_prob", prominence_thresh, cutoff_threshold
    )
    marked_peaks_df = marked_peaks_df.query("defect_marker > 0 and id_corro_prob > 0.05")
    if len(marked_peaks_df) == 0:
        return None
    marked_peaks_df = grouper(marked_peaks_df)
    if len(marked_peaks_df) == 0:
        return None
    return marked_peaks_df


def main() -> None:
    files_track = sorted(
        glob.glob(
            "/home/sayakdg/ILIWorkflows/workflows/corrosion_detection/"
            "data-files/0AXVHCA77S1/corrosion_detection/corrosion_v4.2/*/*/0/*parquet"
        )
    )

    parts = []
    defect_id_offset = 0

    with ProcessPoolExecutor() as pool:
        results = list(
            tqdm(
                pool.map(process_file, files_track),
                total=len(files_track),
            )
        )

    for marked_peaks_df in results:
        if marked_peaks_df is None:
            continue
        marked_peaks_df = marked_peaks_df.copy()
        marked_peaks_df["defect_marker"] += defect_id_offset
        defect_id_offset = int(marked_peaks_df["defect_marker"].max())
        parts.append(marked_peaks_df)

    defectDF = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    defectDF = defectDF.rename(
        columns={
            "id_corro_prob_count": "length",
            "id_corro_prob_max": "vol_conf",
        }
    )

    # markers = np.arange(0.1, 1, 0.05)  # includes 1.0; no rounding
    # defectDF["bin_size"] = np.round(nearest_marker(defectDF["vol_conf"].to_numpy(),markers),2)

    edges = np.arange(0.1, 1.0 + 1e-9, 0.05)  # include top edge
    defectDF["bin_size"] = pd.cut(
        defectDF["vol_conf"],
        bins=edges,
        right=False,  # [0.10, 0.15), [0.15, 0.20), ...
        labels=edges[:-1],
    ).astype(float)

    defectDF.to_parquet("peaks_grouping.parquet", index=False)


if __name__ == "__main__":
    main()
