"""Simple dashboard for the AE keep pipeline.

Calls ``process_detections`` from ``process_ae_detections`` (same keep
rules as the notebook) and shows the log-IQR test plus the booked-track
histogram after keep + ``filter_close_detections``.

Streamlit note: this file is not a server with callbacks. Each widget
click reruns the script top-to-bottom. ``st.session_state`` is how you
keep results across those reruns (same pattern as a form that submits
a query and then keeps showing the last table).

Run:
    /home/sayakdg/myPython312Env/ml_env/bin/streamlit run dashboard_ae_detections.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from process_ae_detections import process_detections

DEFAULT_RESULTS_DIR = "/home/sayakdg/stats-experiments/dent_ae_v4.0"


# ---------------------------------------------------------------------------
# Plots (drawn only after a successful Run)
# ---------------------------------------------------------------------------

def plot_log_iqr(info):
    """Notebook log-IQR test: number line of log(counts) + raw-count bars."""
    log_c = np.asarray(info["iqr_log_counts"], dtype=float)
    tracks = np.arange(len(log_c))
    ok = np.isfinite(log_c)
    exclude = set(info["iqr_exclude"])
    q1, q3, fence = info["iqr_q1"], info["iqr_q3"], info["iqr_upper_fence"]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))

    # Left: each track as a point on log(win count). Red = above the fence.
    ax = axes[0]
    ax.scatter(log_c[ok], np.zeros(ok.sum()), s=22, zorder=3)
    # Stagger labels by sorted log so nearby tracks do not sit on the same row.
    levels = (0.05, 0.13, -0.07, -0.16)
    order = tracks[ok][np.argsort(log_c[ok])]
    dy_of = {int(t): levels[i % len(levels)] for i, t in enumerate(order)}
    for t in tracks[ok]:
        color = "C3" if t in exclude else "k"
        if t in exclude:
            ax.scatter(log_c[t], 0, s=50, color="C3", zorder=4)
        dy = dy_of[int(t)]
        ax.annotate(
            str(t),
            (log_c[t], dy),
            ha="center",
            va="bottom" if dy > 0 else "top",
            fontsize=7,
            color=color,
        )
    if np.isfinite(q1) and np.isfinite(q3):
        ax.axvline(q1, color="C0", ls="--", label=f"Q1={q1:.2f}")
        ax.axvline(q3, color="C0", ls="--", label=f"Q3={q3:.2f}")
        ax.axvspan(q1, q3, color="C0", alpha=0.15, label="IQR box")
    if np.isfinite(fence):
        ax.axvline(
            fence,
            color="C3",
            ls=":",
            label=f"fence={fence:.2f}  (~{np.exp(fence):.0f} wins)",
        )
    ax.set_yticks([])
    ax.set_ylim(-0.24, 0.22)
    ax.set_xlabel("log(win count)")
    ax.set_title("log-IQR test")
    ax.legend(loc="upper left", fontsize=6, framealpha=0.9)

    # Right: same fence on the raw win-count scale (exp of the log values).
    ax = axes[1]
    colors = ["C3" if i in exclude else "C0" for i in tracks]
    ax.bar(tracks, np.exp(np.nan_to_num(log_c)), color=colors)
    if np.isfinite(fence):
        ax.axhline(np.exp(fence), color="C3", ls=":", label=f"fence ≈ {np.exp(fence):.0f}")
        ax.legend(fontsize=7)
    ax.set_xticks(tracks)
    ax.set_xlabel("track")
    ax.set_ylabel("win count")
    ax.set_title("same cutoff on raw counts")
    fig.tight_layout()
    return fig


def plot_keep(final_df, num_tracks):
    """Histogram of who was booked after keep + filter_close_detections."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    counts, _, patches = ax.hist(
        final_df["max_prob_track"],
        bins=num_tracks,
        range=(-0.5, num_tracks - 0.5),
        align="mid",
        color="C0",
        edgecolor="white",
    )
    for count, patch in zip(counts, patches):
        if count > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                count,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(range(num_tracks))
    ax.set_xlabel("max_prob_track")
    ax.set_ylabel("windows")
    ax.set_title("Keep + filter_close_detections: booked track")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit page
#
# Generic pattern: widgets collect inputs → a button triggers work →
# results live in session_state → the rest of the script renders them.
# Same shape as "CSV path + Run analysis" or "SQL query + Submit".
# ---------------------------------------------------------------------------

# Must be the first Streamlit call. Sets browser tab title and layout.
# Generic: any app that needs a wide table/plot page uses layout="wide".
st.set_page_config(page_title="AE detections", layout="wide")
st.title("AE dent keep")

# st.text_input: one-line string. Generic use: file path, API URL, search box.
# ``value=`` is the default the first time; later reruns keep what the user typed.
results_dir = st.text_input("Data directory", value=DEFAULT_RESULTS_DIR)

# st.number_input: numeric spinner. Generic use: batch size, k, max rows.
num_tracks = st.number_input("Number of tracks", min_value=1, value=22, step=1)

# st.button is True only on the rerun caused by that click, then False again.
# Generic use: "Train", "Export", "Refresh" — do the expensive work only here.
if st.button("Run", type="primary"):
    path = Path(results_dir).expanduser()
    if not path.is_dir():
        # st.error: red banner. Generic use: bad path, failed API, missing file.
        st.error(f"Not a directory: {path}")
    else:
        # st.spinner: "please wait" overlay while a long call runs.
        # Generic use: model inference, DB query, downloading a file.
        with st.spinner("Running process_detections…"):
            _, _, final_df, info = process_detections(
                results_dir=str(path),
                num_tracks=int(num_tracks),
            )
        # session_state is a dict that survives reruns in this browser tab.
        # Generic use: cache the last dataframe so changing a slider does not
        # re-hit the database; or store "logged_in", "selected_id", etc.
        st.session_state["final_df"] = final_df
        st.session_state["info"] = info
        st.session_state["num_tracks"] = int(num_tracks)

# Read last run. .get() is None before the first successful Run.
# Generic: same as "show cached results if we have them, else a prompt".
info = st.session_state.get("info")
final_df = st.session_state.get("final_df")
n_tracks = st.session_state.get("num_tracks", int(num_tracks))

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if info is None:
    # st.info: blue hint. Generic use: empty state, "upload a file first".
    st.info("Set the data path and press Run.")
else:
    # st.columns: side-by-side layout. Generic use: KPI row, two plots, form | preview.
    c1, c2, c3, c4 = st.columns(4)
    # st.metric: big number tile. Generic use: revenue, error rate, n_rows.
    c1.metric("files", info["n_files"])
    c2.metric("all_dfs", info["n_all"])
    c3.metric("filtered", info["n_filtered"])
    c4.metric("final", info["n_final"])

    # st.write: markdown / objects. Generic use: status line, a small table.
    st.write(f"IQR exclude: `{info['iqr_exclude']}`")
    # Per-IQR-track keep bar = that track's own 99th percentile of winning max_prob.
    if info["strict_th"]:
        st.write(
            "Strict bars: "
            + ", ".join(
                f"track {t} = {bar:.4f}" for t, bar in info["strict_th"].items()
            )
        )

    # st.pyplot: show a matplotlib Figure. width="content" = do not stretch
    # to the page. Generic use: any existing matplotlib plot in a dashboard.
    st.pyplot(plot_log_iqr(info), clear_figure=True, width="content")
    # st.caption: small text under a figure. Generic use: axis units, source.
    st.caption(
        "Log-IQR test: take log(win count) for every track that won at least once, "
        "then flag anything above **Q3 + 1.5×IQR** (Tukey's upper fence). "
        "On a normal that fence is about +2.70σ / the 99.65th percentile one-sided. "
        "Red = flagged."
    )
    st.pyplot(plot_keep(final_df, n_tracks), clear_figure=True, width="content")
