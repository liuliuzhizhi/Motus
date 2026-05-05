"""
============================================================
Arising-from-chair  Step 3 / 5 - Stand-up event detection
============================================================
Backend usage:
    from step3_arising_event_detection import detect_arising_events

    events_df = detect_arising_events(signal_df, fps=30)

`events_df` columns:
    event_id, category, rise_start_time, rise_end_time, rise_amplitude,
    stand_speed, stand_start_time, stand_end_time, stand_duration

Categories:  "Successful" | "Unsuccessful"
Empty DataFrame (with these columns) is returned when the signal amplitude
is too small to contain any rise.
============================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# Headless-safe matplotlib for the optional diagnostic figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# Tunable thresholds (kept identical to the original script)
# ============================================================
FALL_THRESHOLD              = 0.30
VELOCITY_THRESHOLD          = 0.05
POST_CHECK_RANGE            = 0.30
MIN_RISE                    = 0.08
SUCCESS_RISE                = 0.25
LOW_AMPLITUDE_SKIP          = 0.25     # below this, no events at all

EMPTY_EVENT_COLUMNS = [
    "event_id", "category",
    "rise_start_time", "rise_end_time", "rise_amplitude",
    "stand_speed",
    "stand_start_time", "stand_end_time", "stand_duration",
]

PALETTE = {
    "curve":   "#2E4057",
    "failed":  "#E07A5F",
    "success": "#4F8C5A",
    "standing":"#6FA8DC",
    "grid":    "#E6E6E6",
    "panel":   "#FAFAFA",
}


# ============================================================
# Core algorithm helpers (logic preserved verbatim)
# ============================================================
def _post_trim_plateau(x, seg_start, seg_peak, fps,
                       velocity_threshold=VELOCITY_THRESHOLD,
                       post_check_range=POST_CHECK_RANGE):
    x_seg = x[seg_start:seg_peak + 1]
    if len(x_seg) < 5:
        return seg_peak
    v = np.diff(x_seg)
    v = gaussian_filter1d(v, sigma=2)
    max_v = np.max(np.abs(v))
    if max_v == 0:
        return seg_peak
    threshold = velocity_threshold * max_v
    check_start = int(post_check_range * len(v))
    consecutive = 0
    for i in range(check_start, len(v)):
        if abs(v[i]) < threshold:
            consecutive += 1
            if consecutive >= 3:
                return seg_start + i - 2
        else:
            consecutive = 0
    return seg_peak


def _pre_trim_plateau(x, seg_start, seg_peak,
                      velocity_threshold=VELOCITY_THRESHOLD):
    if seg_peak - seg_start < 5:
        return seg_start
    x_seg = x[seg_start:seg_peak + 1]
    v = np.diff(x_seg)
    v = gaussian_filter1d(v, sigma=2)
    max_v = np.max(v)
    if max_v <= 0:
        return seg_start
    threshold = velocity_threshold * max_v
    max_idx = np.argmax(v)
    new_start = None
    for k in range(max_idx, -1, -1):
        if np.all(v[k:max_idx + 1] > threshold):
            new_start = k
        else:
            break
    return seg_start + new_start if new_start is not None else seg_start


def _find_rising_segments(distance, fps,
                          fall_threshold=FALL_THRESHOLD,
                          velocity_threshold=VELOCITY_THRESHOLD,
                          post_check_range=POST_CHECK_RANGE,
                          min_rise=MIN_RISE):
    """Identify monotonic-ish rising segments and trim plateau ends."""
    x = np.asarray(distance, dtype=float)
    n = len(x)
    segments = []
    if n < 2:
        return segments

    seg_start_idx = 0
    seg_peak_idx  = 0
    seg_peak_val  = x[0]

    def emit(seg_start, seg_peak, peak_val):
        rise = peak_val - x[seg_start]
        if rise >= min_rise:
            segments.append({
                "start_idx": seg_start,
                "peak_idx":  seg_peak,
                "rise":      float(rise),
                "start_val": float(x[seg_start]),
                "peak_val":  float(peak_val),
            })

    i = 1
    while i < n:
        if x[i] >= x[i - 1]:
            if x[i] > seg_peak_val:
                seg_peak_val = x[i]
                seg_peak_idx = i
            i += 1
            continue

        descent_top_val = x[i - 1]
        j = i
        while j < n and x[j] < x[j - 1]:
            j += 1
        descent_bottom_idx = j - 1
        descent_bottom_val = x[descent_bottom_idx]

        descent_mag = descent_top_val - descent_bottom_val
        current_rise = seg_peak_val - x[seg_start_idx]

        if current_rise > 0 and descent_mag > fall_threshold * current_rise:
            seg_start_idx = _pre_trim_plateau(x, seg_start_idx, seg_peak_idx,
                                              velocity_threshold)
            seg_peak_idx  = _post_trim_plateau(x, seg_start_idx, seg_peak_idx, fps,
                                               velocity_threshold, post_check_range)
            seg_peak_val  = x[seg_peak_idx]
            emit(seg_start_idx, seg_peak_idx, seg_peak_val)
            seg_start_idx = descent_bottom_idx
            seg_peak_idx  = descent_bottom_idx
            seg_peak_val  = descent_bottom_val
        elif current_rise == 0:
            seg_start_idx = descent_bottom_idx
            seg_peak_idx  = descent_bottom_idx
            seg_peak_val  = descent_bottom_val
        i = j

    seg_start_idx = _pre_trim_plateau(x, seg_start_idx, seg_peak_idx,
                                      velocity_threshold)
    seg_peak_idx  = _post_trim_plateau(x, seg_start_idx, seg_peak_idx, fps,
                                       velocity_threshold, post_check_range)
    seg_peak_val  = x[seg_peak_idx]
    emit(seg_start_idx, seg_peak_idx, seg_peak_val)

    return segments


def _extract_events(time, distance, segments,
                    success_rise=SUCCESS_RISE):
    """Turn rising segments into event records with rise/stand metrics."""
    events = []
    n = len(distance)

    for i, seg in enumerate(segments):
        rise_total = seg["rise"]
        start_idx  = seg["start_idx"]
        peak_idx   = seg["peak_idx"]
        start_val  = seg["start_val"]
        peak_val   = seg["peak_val"]

        category = "Successful" if rise_total >= success_rise else "Unsuccessful"

        rise_10 = start_val + 0.10 * rise_total
        rise_95 = start_val + 0.95 * rise_total
        segment_slice = distance[start_idx:peak_idx + 1]
        idx_10 = start_idx + int(np.argmin(np.abs(segment_slice - rise_10)))
        idx_95 = start_idx + int(np.argmin(np.abs(segment_slice - rise_95)))

        ev = {
            "event_id":        i + 1,
            "category":        category,
            "rise_start_time": float(time[idx_10]),
            "rise_end_time":   float(time[idx_95]),
            "rise_amplitude":  float(distance[idx_95] - distance[idx_10]),
            "stand_speed":     None,
            "stand_start_time": None,
            "stand_end_time":   None,
            "stand_duration":   None,
        }

        if category == "Successful":
            dt = time[idx_95] - time[idx_10]
            stand_speed = (distance[idx_95] - distance[idx_10]) / dt if dt > 0 else 0.0
            threshold = peak_val - 0.10 * rise_total
            stand_end_idx = None
            for k in range(peak_idx, n):
                if distance[k] <= threshold:
                    stand_end_idx = k
                    break
            if stand_end_idx is None:
                stand_end_idx = n - 1
            ev["stand_speed"]      = float(stand_speed)
            ev["stand_start_time"] = float(time[idx_95])
            ev["stand_end_time"]   = float(time[stand_end_idx])
            ev["stand_duration"]   = float(time[stand_end_idx] - time[idx_95])

        events.append(ev)
    return events


# ============================================================
# Plotting
# ============================================================
def _style_axis(ax):
    ax.set_facecolor(PALETTE["panel"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _plot_assessment(time, distance, segments, events, save_path, title):
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    _style_axis(ax)
    ax.plot(time, distance, color=PALETTE["curve"], linewidth=1.6)

    y_level = 0.92; y_gap = 0.06
    for i, seg in enumerate(segments):
        x_start = time[seg["start_idx"]]
        x_end   = time[seg["peak_idx"]]
        current_y = y_level - i * y_gap
        ax.axvline(x_start, linestyle="--", linewidth=1)
        ax.axvline(x_end,   linestyle="--", linewidth=1)
        ax.annotate("", xy=(x_end, current_y), xytext=(x_start, current_y),
                    xycoords=ax.get_xaxis_transform(),
                    arrowprops=dict(arrowstyle="<->", linewidth=1.5))
        ax.text((x_start + x_end) / 2, current_y + 0.02, f"Rising {i+1}",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom")

    for ev in events:
        color = PALETTE["success"] if ev["category"] == "Successful" else PALETTE["failed"]
        ax.axvspan(ev["rise_start_time"], ev["rise_end_time"],
                   alpha=0.30, facecolor=color)
        if ev["category"] == "Successful":
            ax.axvspan(ev["stand_start_time"], ev["stand_end_time"],
                       alpha=0.15, facecolor=PALETTE["standing"])

    handles = []
    if any(ev["category"] == "Successful" for ev in events):
        handles.append(Patch(facecolor=PALETTE["success"], alpha=0.30, label="Successful Stand-up"))
    if any(ev["category"] == "Unsuccessful" for ev in events):
        handles.append(Patch(facecolor=PALETTE["failed"],  alpha=0.30, label="Unsuccessful Attempt"))
    if any(ev["category"] == "Successful" for ev in events):
        handles.append(Patch(facecolor=PALETTE["standing"], alpha=0.15, label="Standing Duration"))
    if handles:
        ax.legend(handles=handles, loc="best", frameon=True)

    ax.set_title("Precisely Localized Stand-up Events")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Normalized Distance")
    fig.suptitle(title)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_low_amplitude(time, distance, save_path, title):
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    _style_axis(ax)
    ax.plot(time, distance, color=PALETTE["curve"], linewidth=1.6)
    ymax = float(np.max(distance)); ymin = float(np.min(distance))
    amplitude = ymax - ymin
    ax.axhline(ymax, linestyle="--", linewidth=1.2)
    ax.axhline(ymin, linestyle="--", linewidth=1.2)
    x_mid = time[len(time) // 2]
    ax.annotate("", xy=(x_mid, ymax), xytext=(x_mid, ymin),
                arrowprops=dict(arrowstyle="<->", linewidth=1.5))
    ax.text(x_mid, ymin + amplitude / 2,
            f"Amplitude = {amplitude:.3f}", ha="left", va="center", fontsize=11)
    ax.set_title("Low-Amplitude Curve (No Valid Stand-up Attempt)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Normalized Distance")
    fig.suptitle(title)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main entry
# ============================================================
def detect_arising_events(signal_df,
                          fps=30,
                          save_plot_path=None,
                          plot_title="Parkinson Stand-up Assessment"):
    """
    Run rise-segment detection and event classification on a signal.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Output of step 2. Must contain `time` and `distance` columns.
    fps : float
    save_plot_path : str or None
    plot_title : str
        Used in the optional diagnostic figure suptitle.

    Returns
    -------
    pd.DataFrame   (possibly empty)
    """
    if signal_df is None or len(signal_df) == 0:
        return pd.DataFrame(columns=EMPTY_EVENT_COLUMNS)
    for c in ("time", "distance"):
        if c not in signal_df.columns:
            raise ValueError(f"signal_df missing column: {c}")

    time = pd.to_numeric(signal_df["time"], errors="coerce").to_numpy()
    distance = pd.to_numeric(signal_df["distance"], errors="coerce").to_numpy()
    mask = ~(np.isnan(time) | np.isnan(distance))
    time = time[mask]; distance = distance[mask]

    if len(distance) < 3:
        return pd.DataFrame(columns=EMPTY_EVENT_COLUMNS)

    global_amplitude = float(np.max(distance) - np.min(distance))

    # Low-amplitude short-circuit
    if global_amplitude < LOW_AMPLITUDE_SKIP:
        print(f"[arising] amplitude {global_amplitude:.3f} < {LOW_AMPLITUDE_SKIP}"
              f" -> no events")
        if save_plot_path:
            _plot_low_amplitude(time, distance, save_plot_path, plot_title)
        return pd.DataFrame(columns=EMPTY_EVENT_COLUMNS)

    segments = _find_rising_segments(distance, fps=fps)
    events   = _extract_events(time, distance, segments)

    if save_plot_path:
        _plot_assessment(time, distance, segments, events,
                         save_plot_path, plot_title)

    return pd.DataFrame(events, columns=EMPTY_EVENT_COLUMNS)


# ============================================================
# >>>>>>>>>>>>>>>>>>  DEBUG / DEV-ONLY  >>>>>>>>>>>>>>>>>>>>>>
# Edit the paths below and run:
#   python step3_arising_event_detection.py
# Delete this whole section for production - nothing above depends on it.
# ============================================================
if __name__ == "__main__":
    DEBUG_SIGNAL_CSV = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\h_arising-from-chair\b_kinematic-signal\debug_kinematic_signal_norm.csv"
    DEBUG_OUTPUT_PLOT = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\h_arising-from-chair\c_action-recognition\arising_event_detection.png"
    DEBUG_OUTPUT_CSV = DEBUG_OUTPUT_PLOT.replace(".png", ".csv")
    DEBUG_FPS = 30

    print(f"[DEBUG] Reading signal CSV: {DEBUG_SIGNAL_CSV}")
    sig = pd.read_csv(DEBUG_SIGNAL_CSV)

    print(f"[DEBUG] Detecting arising events (fps={DEBUG_FPS})")
    events = detect_arising_events(sig, fps=DEBUG_FPS,
                                   save_plot_path=DEBUG_OUTPUT_PLOT,
                                   plot_title="DEBUG arising-from-chair")
    print(f"[DEBUG] Got {len(events)} events")
    print(events.to_string(index=False) if len(events) else "[DEBUG] (no events)")

    os.makedirs(os.path.dirname(DEBUG_OUTPUT_CSV) or ".", exist_ok=True)
    events.to_csv(DEBUG_OUTPUT_CSV, index=False)
    print(f"[DEBUG] Events CSV : {DEBUG_OUTPUT_CSV}")
    print(f"[DEBUG] Plot       : {DEBUG_OUTPUT_PLOT}")
# ============================================================
# <<<<<<<<<<<<<<<<<<  END DEBUG / DEV-ONLY  <<<<<<<<<<<<<<<<<<
# ============================================================