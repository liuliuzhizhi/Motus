"""
============================================================
Arising-from-chair  Step 2 / 5 - Kinematic signal (single patient)
============================================================
Backend usage:
    from step2_arising_kinematic_signal import compute_arising_signal

    signal_df = compute_arising_signal(keypoints_df, fps=30)

`signal_df` columns: frame, time, distance, speed
============================================================
"""
import os
import numpy as np
import pandas as pd

REQUIRED_KEYPOINT_COLS = ("frame", "landmark_id", "x", "y")


def compute_arising_signal(keypoints_df, fps=30,
                           save_csv_path=None,
                           save_plot_path=None):
    """
    Build the body-rise signal used for arising-from-chair detection.

    The signal is the vertical distance between the mean ankle position
    (landmarks 27, 28) and the mean shoulder position (landmarks 11, 12),
    self-normalised so the deepest squat sits at 1.0.

    Parameters
    ----------
    keypoints_df : pd.DataFrame
        Columns: frame, landmark_id, x, y. Output of step 1.
    fps : float
    save_csv_path, save_plot_path : str or None
        If provided, also persist the CSV / quick-look PNG.

    Returns
    -------
    pd.DataFrame with columns frame, time, distance, speed.
    Empty DataFrame if the required landmarks are missing.
    """
    if keypoints_df is None or len(keypoints_df) == 0:
        return pd.DataFrame(columns=["frame", "time", "distance", "speed"])

    missing = [c for c in REQUIRED_KEYPOINT_COLS if c not in keypoints_df.columns]
    if missing:
        raise ValueError(f"keypoints_df missing columns: {missing}")

    df = keypoints_df[["frame", "landmark_id", "x", "y"]]

    df_11 = df[df["landmark_id"] == 11].rename(columns={"x": "x11", "y": "y11"})
    df_12 = df[df["landmark_id"] == 12].rename(columns={"x": "x12", "y": "y12"})
    df_27 = df[df["landmark_id"] == 27].rename(columns={"x": "x27", "y": "y27"})
    df_28 = df[df["landmark_id"] == 28].rename(columns={"x": "x28", "y": "y28"})

    merged_scale = pd.merge(
        df_11[["frame", "x11", "y11"]],
        df_12[["frame", "x12", "y12"]],
        on="frame", how="inner",
    )
    merged = pd.merge(
        df_27[["frame", "y27"]],
        df_28[["frame", "y28"]],
        on="frame", how="inner",
    )
    merged = pd.merge(merged, merged_scale, on="frame", how="inner")

    if merged.empty:
        return pd.DataFrame(columns=["frame", "time", "distance", "speed"])

    # Optional torso-length scale (unused below but kept for future tweaks)
    merged["scale_11_12"] = np.sqrt(
        (merged["x11"] - merged["x12"]) ** 2 +
        (merged["y11"] - merged["y12"]) ** 2
    )

    # Body rise = ankles - shoulders, in image y. Shoulders rise => y goes
    # smaller, so we negate to make standing-up a positive direction.
    merged["distance"] = (
        -(merged["y11"] + merged["y12"]) / 2
        + (merged["y27"] + merged["y28"]) / 2
    )

    # Self-normalise. The original script divides by .min() (a NEGATIVE number),
    # which flips the sign so that "deeply seated" becomes 1.0 and "fully
    # standing" approaches 0. We preserve that exact behaviour to keep the
    # downstream thresholds unchanged.
    min_val = merged["distance"].min()
    if min_val == 0 or not np.isfinite(min_val):
        # Degenerate signal - return empty so downstream stages can short-circuit.
        return pd.DataFrame(columns=["frame", "time", "distance", "speed"])
    merged["distance"] = merged["distance"] / min_val

    merged["time"] = merged["frame"] / fps
    merged["speed"] = np.abs(merged["distance"].diff()) * fps
    if len(merged) > 1:
        merged.loc[merged.index[0], "speed"] = merged.loc[merged.index[1], "speed"]
    else:
        merged.loc[merged.index[0], "speed"] = 0.0

    out = merged[["frame", "time", "distance", "speed"]].reset_index(drop=True)

    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path) or ".", exist_ok=True)
        out.to_csv(save_csv_path, index=False)

    if save_plot_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(save_plot_path) or ".", exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.plot(out["time"], out["distance"])
        plt.xlabel("Time (s)")
        plt.ylabel("Vertical distance (normalised)")
        plt.title(os.path.splitext(os.path.basename(save_plot_path))[0])
        plt.grid(True)
        plt.savefig(save_plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    return out


# ============================================================
# >>>>>>>>>>>>>>>>>>  DEBUG / DEV-ONLY  >>>>>>>>>>>>>>>>>>>>>>
# Edit the paths below and run:  python step2_arising_kinematic_signal.py
# Delete this whole section for production - nothing above depends on it.
# ============================================================
if __name__ == "__main__":
    DEBUG_KEYPOINT_CSV = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\h_arising-from-chair\a_landmark-detection\Patient100_arising-from-chair_1\Patient100_arising-from-chair_1_keypoints.csv"
    DEBUG_OUTPUT_CSV  = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\h_arising-from-chair\b_kinematic-signal\debug_kinematic_signal_norm.csv"
    DEBUG_OUTPUT_PLOT = DEBUG_OUTPUT_CSV.replace(".csv", "_plot.png")
    DEBUG_FPS = 30

    print(f"[DEBUG] Reading keypoints: {DEBUG_KEYPOINT_CSV}")
    keypoints = pd.read_csv(DEBUG_KEYPOINT_CSV)

    print(f"[DEBUG] Computing arising signal (fps={DEBUG_FPS})")
    out = compute_arising_signal(
        keypoints, fps=DEBUG_FPS,
        save_csv_path=DEBUG_OUTPUT_CSV,
        save_plot_path=DEBUG_OUTPUT_PLOT,
    )
    print(f"[DEBUG] Got {len(out)} rows")
    print(f"[DEBUG] CSV  : {DEBUG_OUTPUT_CSV}")
    print(f"[DEBUG] Plot : {DEBUG_OUTPUT_PLOT}")
# ============================================================
# <<<<<<<<<<<<<<<<<<  END DEBUG / DEV-ONLY  <<<<<<<<<<<<<<<<<<
# ============================================================