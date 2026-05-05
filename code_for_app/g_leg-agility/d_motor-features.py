"""
============================================================
Step 4 / 5 - Kinematic feature extraction (single patient)
============================================================
Backend usage:
    from step4_feature_extraction import extract_features

    features = extract_features(events_df, fps=30, max_events=12)
    # features is a JSON-serialisable dict with 18 keys.

The output dict is ready to be fed to step5_rule_scoring.score_from_features().
============================================================
"""
import os
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Helpers (unchanged from original)
# ------------------------------------------------------------
def safe_cv(values):
    """Coefficient of variation = std / mean. NaN if mean is 0 or empty."""
    if values is None or len(values) == 0:
        return np.nan
    mean = np.mean(values)
    if mean == 0:
        return np.nan
    return float(np.std(values, ddof=0) / mean)


def find_first_decay(amplitudes):
    """Position of the first 3-event run that satisfies the decay criterion."""
    for i in range(len(amplitudes) - 2):
        a1, a2, a3 = amplitudes[i], amplitudes[i + 1], amplitudes[i + 2]
        cond_a = (a2 < 0.9 * a1) and (a3 < 0.9 * a2)
        cond_b = (a2 < 0.8 * a1) and (a3 < 0.8 * a1)
        if cond_a or cond_b:
            return i
    return None


# ------------------------------------------------------------
# Main entry
# ------------------------------------------------------------
FEATURE_KEYS = [
    "mean_amplitude", "median_amplitude", "max_amplitude",
    "mean_duration", "median_duration", "median_movement_rate",
    "amplitude_slope", "duration_slope",
    "first_decay_position", "decay_occurred",
    "amplitude_cv", "duration_cv", "max_duration",
    "interval_cv", "max_interval",
    "num_pauses", "num_freezes", "num_events",
]


def extract_features(events_df, fps=30.0, max_events=12):
    """
    Compute the 18 kinematic features described in the project notes.

    Parameters
    ----------
    events_df : pd.DataFrame
        Output of step 3. Empty DataFrame is allowed - all features will
        be NaN except `num_events` which becomes 0.
    fps : float
    max_events : int
        Only the first `max_events` events drive features 1-4. Feature
        `num_events` always reflects the full count.

    Returns
    -------
    dict   (JSON-serialisable; NaN values become Python NaN floats)
    """
    features = {k: np.nan for k in FEATURE_KEYS}
    features["num_events"] = 0

    if events_df is None or len(events_df) == 0:
        return features

    features["num_events"] = int(len(events_df))

    df = events_df.iloc[:max_events].copy()
    if len(df) == 0:
        return features

    amplitudes    = df["peak_amplitude"].to_numpy(dtype=float)
    durations     = df["duration_frames"].to_numpy(dtype=float) / fps  # 换算成秒
    start_frames  = df["start_frame"].to_numpy(dtype=float)
    end_frames    = df["end_frame"].to_numpy(dtype=float)
    rise90_frames = df["rise90_frame"].to_numpy(dtype=float)
    fall90_frames = df["fall90_frame"].to_numpy(dtype=float)
    n = len(df)

    # 1) Hypokinesia
    features["mean_amplitude"]   = float(np.mean(amplitudes))
    features["median_amplitude"] = float(np.median(amplitudes))
    features["max_amplitude"]    = float(np.max(amplitudes))

    # 2) Bradykinesia
    features["mean_duration"]   = float(np.mean(durations))
    features["median_duration"] = float(np.median(durations))

    rise_times = rise90_frames - start_frames
    fall_times = end_frames - fall90_frames
    valid = (rise_times > 0) & (fall_times > 0)
    if valid.any():
        rates = (0.9 * amplitudes[valid] / rise_times[valid] * fps
                 + 0.9 * amplitudes[valid] / fall_times[valid] * fps)
        features["median_movement_rate"] = float(np.median(rates))

    # 3) Sequence effect
    if n >= 2:
        x = np.arange(n, dtype=float)
        features["amplitude_slope"] = float(np.polyfit(x, amplitudes, 1)[0])
        features["duration_slope"]  = float(np.polyfit(x, durations, 1)[0])

    decay_pos = find_first_decay(amplitudes)
    if decay_pos is not None:
        features["first_decay_position"] = int(decay_pos)
        features["decay_occurred"] = 1
    else:
        features["first_decay_position"] = 10
        features["decay_occurred"] = 0

    # 4) Hesitation-halts
    features["amplitude_cv"] = safe_cv(amplitudes)
    features["duration_cv"]  = safe_cv(durations)
    features["max_duration"] = float(np.max(durations))

    intervals = np.array([])
    if n >= 2:
        intervals = (start_frames[1:] - end_frames[:-1]) / fps  #换算成秒
        features["interval_cv"]  = safe_cv(intervals)
        features["max_interval"] = float(np.max(intervals)) / fps if len(intervals) > 0 else np.nan

    median_dur = float(np.median(durations))
    median_int = float(np.median(intervals)) if len(intervals) > 0 else 0.0
    pause_count = 0
    freeze_count = 0
    if median_dur > 0:
        pause_count  += int(np.sum(durations > 2 * median_dur))
        freeze_count += int(np.sum(durations > 4 * median_dur))
    if len(intervals) > 0 and median_int > 0:
        pause_count  += int(np.sum(intervals > 2 * median_int))
        freeze_count += int(np.sum(intervals > 4 * median_int))
    features["num_pauses"]  = pause_count
    features["num_freezes"] = freeze_count

    return features


# ============================================================
# >>>>>>>>>>>>>>>>>>  DEBUG / DEV-ONLY  >>>>>>>>>>>>>>>>>>>>>>
# Edit the paths below and run:  python step4_feature_extraction.py
# Delete this whole section for production - nothing above depends on it.
# ============================================================
if __name__ == "__main__":
    import json
    DEBUG_EVENT_CSV = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\leg-agility\c_event-detection\debug_events.csv"
    DEBUG_OUTPUT_JSON = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\leg-agility\d_rate\kinematic_features\debug_features.json"
    DEBUG_FPS = 30.0

    print(f"[DEBUG] Reading events CSV: {DEBUG_EVENT_CSV}")
    try:
        events = pd.read_csv(DEBUG_EVENT_CSV)
    except pd.errors.EmptyDataError:
        events = pd.DataFrame()

    print(f"[DEBUG] Extracting features (fps={DEBUG_FPS})")
    features = extract_features(events, fps=DEBUG_FPS)

    # Convert NaN -> None for JSON
    serialisable = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                    for k, v in features.items()}
    print(f"[DEBUG] Features:")
    print(json.dumps(serialisable, indent=2, ensure_ascii=False))

    if DEBUG_OUTPUT_JSON:
        os.makedirs(os.path.dirname(DEBUG_OUTPUT_JSON) or ".", exist_ok=True)
        with open(DEBUG_OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, ensure_ascii=False, indent=2)
        print(f"[DEBUG] Wrote features JSON: {DEBUG_OUTPUT_JSON}")
# ============================================================
# <<<<<<<<<<<<<<<<<<  END DEBUG / DEV-ONLY  <<<<<<<<<<<<<<<<<<
# ============================================================