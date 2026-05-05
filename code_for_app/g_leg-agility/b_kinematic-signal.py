"""
============================================================
Step 2 / 5 - Build the kinematic signal (single patient)
============================================================
Backend usage:
    from step2_kinematic_signal import compute_kinematic_signal

    kinematic_df = compute_kinematic_signal(
        keypoints_df,                # output of step 1
        side="left",                 # or "right"
        fps=30,
    )

`kinematic_df` columns: frame, time, distance, speed
============================================================
"""
import os
import numpy as np
import pandas as pd


REQUIRED_KEYPOINT_COLS = ("frame", "landmark_id", "x", "y")


def compute_kinematic_signal(keypoints_df, side="left", fps=30,
                             save_csv_path=None,
                             save_plot_path=None):
    """
    Convert MediaPipe keypoints into a normalised vertical-distance signal
    between the two ankles, plus its first-derivative speed.

    Parameters
    ----------
    keypoints_df : pd.DataFrame
        Must contain columns: frame, landmark_id, x, y.
        landmark_id 11/12 are shoulders (used for scaling),
        landmark_id 27/28 are ankles.
    side : {"left", "right"}
        Which leg's foot is being lifted - decides the sign of the distance.
    fps : float
    save_csv_path : str or None
        Optional path to write the resulting kinematic CSV.
    save_plot_path : str or None
        Optional path to write a quick distance-vs-time PNG.

    Returns
    -------
    pd.DataFrame with columns: frame, time, distance, speed.
    Returns an empty DataFrame if the inputs do not contain the required
    landmarks.
    """
    if keypoints_df is None or len(keypoints_df) == 0:
        return pd.DataFrame(columns=["frame", "time", "distance", "speed"])

    missing = [c for c in REQUIRED_KEYPOINT_COLS if c not in keypoints_df.columns]
    if missing:
        raise ValueError(f"keypoints_df missing columns: {missing}")

    side = side.lower()
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    df = keypoints_df[["frame", "landmark_id", "x", "y"]]

    # Pull the four landmarks we need
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

    # Shoulder distance used as a per-frame scale (handles camera distance changes)
    merged["scale_11_12"] = np.sqrt(
        (merged["x11"] - merged["x12"]) ** 2 +
        (merged["y11"] - merged["y12"]) ** 2
    )

    # Foot-up direction: image y goes down, so a higher foot has smaller y
    if side == "right":
        merged["distance"] = -(merged["y28"] - merged["y27"])
    else:  # left
        merged["distance"] = -(merged["y27"] - merged["y28"])

    # Express as a percentage of the shoulder distance
    merged["distance"] = merged["distance"] / merged["scale_11_12"] * 100

    merged["time"] = merged["frame"] / fps

    # Speed = absolute first difference, scaled to per-second
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
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(save_plot_path) or ".", exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.plot(out["time"], out["distance"])
        plt.xlabel("Time (s)")
        plt.ylabel("Vertical distance (% of shoulder width)")
        plt.title(os.path.splitext(os.path.basename(save_plot_path))[0])
        plt.grid(True)
        plt.savefig(save_plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    return out


# ============================================================
# >>>>>>>>>>>>>>>>>>  DEBUG / DEV-ONLY  >>>>>>>>>>>>>>>>>>>>>>
# Edit the paths below and run:  python step2_kinematic_signal.py
# Delete this whole section for production - nothing above depends on it.
# ============================================================
if __name__ == "__main__":
    DEBUG_KEYPOINT_CSV = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\leg-agility\a_landmark-detection\Patient1_leg-agility-L_1\Patient1_leg-agility-L_1_keypoints.csv"
    DEBUG_OUTPUT_CSV = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\leg-agility\b_kinematic-signal\debug_kinematic_signal_norm.csv"
    DEBUG_OUTPUT_PLOT = DEBUG_OUTPUT_CSV.replace(".csv", "_plot.png")
    DEBUG_SIDE = "left"
    DEBUG_FPS = 30

    print(f"[DEBUG] Reading keypoints: {DEBUG_KEYPOINT_CSV}")
    keypoints = pd.read_csv(DEBUG_KEYPOINT_CSV)

    print(f"[DEBUG] Computing kinematic signal (side={DEBUG_SIDE}, fps={DEBUG_FPS})")
    out = compute_kinematic_signal(
        keypoints,
        side=DEBUG_SIDE, fps=DEBUG_FPS,
        save_csv_path=DEBUG_OUTPUT_CSV,
        save_plot_path=DEBUG_OUTPUT_PLOT,
    )
    print(f"[DEBUG] Got {len(out)} rows")
    print(f"[DEBUG] CSV  : {DEBUG_OUTPUT_CSV}")
    print(f"[DEBUG] Plot : {DEBUG_OUTPUT_PLOT}")
# ============================================================
# <<<<<<<<<<<<<<<<<<  END DEBUG / DEV-ONLY  <<<<<<<<<<<<<<<<<<
# ============================================================