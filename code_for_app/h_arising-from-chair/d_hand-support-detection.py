"""
============================================================
Arising-from-chair  Step 4 / 5 - Hand-support detection
============================================================
Backend usage:
    from step4_hand_support_detection import HandSupportDetector

    detector = HandSupportDetector(fps=30)
    hs_df = detector.detect(events_df, keypoints_df)

`hs_df` columns (one row per Successful event):
    event_id, left_hand_support, right_hand_support, any_hand_support,
    left_v_ratio, left_p_ratio, right_v_ratio, right_p_ratio,
    left_wrist_below_shoulder_norm, right_wrist_below_shoulder_norm,
    left_wrist_to_knee_norm, right_wrist_to_knee_norm, body_dy_norm
============================================================
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
# Constants (preserved from the original)
# ============================================================
LM = {
    "nose": 0,
    "left_shoulder": 11,  "right_shoulder": 12,
    "left_elbow":    13,  "right_elbow":    14,
    "left_wrist":    15,  "right_wrist":    16,
    "left_hip":      23,  "right_hip":      24,
    "left_knee":     25,  "right_knee":     26,
    "left_ankle":    27,  "right_ankle":    28,
}
SKELETON_EDGES = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 31), (27, 29), (29, 31),
    (28, 32), (28, 30), (30, 32),
]
REQUIRED_LMS = [11, 12, 15, 16, 23, 24]

# Decision thresholds
TH_VERT_RATIO    = 0.40   # 手垂直位移占身体中心垂直位移的阈值
TH_PATH_RATIO    = 0.60   # 手移动的路程身体中心路程的阈值
# 两个限制条件，手抬得很高时不算，几乎没站起不算。
TH_RAISED_NORM   = 0.50
MIN_BODY_DY_NORM = 0.1

# How much of the rise window to use (the patient often releases support
# during the second half of the rise, so we focus on the early portion).
DEFAULT_TIME_RANGE = 0.4 #考虑站起时间前1s到站起时间的前0.4范围


# ============================================================
# Numeric helpers (logic preserved verbatim)
# ============================================================
def _smooth_1d(arr, window=5):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n < 3:
        return arr
    w = min(window, n)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return arr
    pad = w // 2
    padded = np.pad(arr, pad, mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(padded, kernel, mode="valid")


def _slice_landmarks(full_df, f0, f1, smooth_window=5):
    df = full_df[(full_df["frame"] >= f0) & (full_df["frame"] <= f1)]
    if df.empty:
        return {}
    common = np.arange(f0, f1 + 1)
    out = {}
    for lm_id, sub in df.groupby("landmark_id"):
        sub = sub.sort_values("frame")
        f = sub["frame"].values.astype(float)
        x = sub["x"].values.astype(float)
        y = sub["y"].values.astype(float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2:
            continue
        f, x, y = f[valid], x[valid], y[valid]
        x_i = np.interp(common, f, x)
        y_i = np.interp(common, f, y)
        out[int(lm_id)] = {
            "frames": common,
            "x": _smooth_1d(x_i, smooth_window),
            "y": _smooth_1d(y_i, smooth_window),
        }
    return out


def _compute_features(L):
    if not all(k in L for k in REQUIRED_LMS):
        return None
    if any(len(L[k]["x"]) < 3 for k in REQUIRED_LMS):
        return None

    sh_x = (L[11]["x"] + L[12]["x"]) / 2.0
    sh_y = (L[11]["y"] + L[12]["y"]) / 2.0
    hp_x = (L[23]["x"] + L[24]["x"]) / 2.0
    hp_y = (L[23]["y"] + L[24]["y"]) / 2.0

    torso = np.sqrt((sh_x - hp_x) ** 2 + (sh_y - hp_y) ** 2)
    torso_length = float(np.median(torso))
    if torso_length < 1e-6:
        torso_length = 1.0

    bc_x = (sh_x + hp_x) / 2.0
    bc_y = (sh_y + hp_y) / 2.0
    body_path = float(np.sum(np.sqrt(np.diff(bc_x) ** 2 + np.diff(bc_y) ** 2)))
    body_dy = float(bc_y[-1] - bc_y[0])
    body_dy_abs = abs(body_dy)
    body_y_sign = float(np.sign(np.mean(hp_y) - np.mean(sh_y)))
    if body_y_sign == 0:
        body_y_sign = 1.0

    feats = {
        "torso_length":   torso_length,
        "body_path_norm": body_path / torso_length,
        "body_dy_norm":   body_dy_abs / torso_length,
        "body_y_sign":    body_y_sign,
    }

    for side, w_id, sh_id, hp_id, k_id in [
        ("left",  15, 11, 23, 25),
        ("right", 16, 12, 24, 26),
    ]:
        if w_id not in L:
            continue
        wx, wy = L[w_id]["x"], L[w_id]["y"]
        w_path = float(np.sum(np.sqrt(np.diff(wx) ** 2 + np.diff(wy) ** 2)))
        w_dy_abs = abs(float(wy[-1] - wy[0]))

        v_ratio = w_dy_abs / body_dy_abs if body_dy_abs > 1e-6 else 1.0
        p_ratio = w_path / body_path if body_path > 1e-6 else 1.0

        sh_y_end = L[sh_id]["y"][-1]
        hp_y_end = L[hp_id]["y"][-1]
        wrist_below_shoulder = (wy[-1] - sh_y_end) * body_y_sign / torso_length
        wrist_below_hip      = (wy[-1] - hp_y_end) * body_y_sign / torso_length

        if k_id in L:
            wrist_to_knee = float(np.hypot(wx[-1] - L[k_id]["x"][-1],
                                           wy[-1] - L[k_id]["y"][-1])) / torso_length
        else:
            wrist_to_knee = float("nan")

        feats[f"{side}_wrist_path_norm"]      = w_path / torso_length
        feats[f"{side}_wrist_dy_norm"]        = w_dy_abs / torso_length
        feats[f"{side}_v_ratio"]              = float(v_ratio)
        feats[f"{side}_p_ratio"]              = float(p_ratio)
        feats[f"{side}_wrist_below_shoulder"] = float(wrist_below_shoulder)
        feats[f"{side}_wrist_below_hip"]      = float(wrist_below_hip)
        feats[f"{side}_wrist_to_knee_norm"]   = float(wrist_to_knee)
        feats[f"{side}_wrist_std_x_norm"]     = float(np.std(wx) / torso_length)
        feats[f"{side}_wrist_std_y_norm"]     = float(np.std(wy) / torso_length)

    return feats


def _decide(feats):
    body_dy_norm = feats.get("body_dy_norm", 0.0)
    body_too_small = body_dy_norm < MIN_BODY_DY_NORM

    decisions = {}
    for side in ("left", "right"):
        if f"{side}_v_ratio" not in feats:
            decisions[side] = {
                "support": False, "reason": "Wrist landmark missing.",
                "criteria": {}, "features": {},
            }
            continue
        v       = feats[f"{side}_v_ratio"]
        p       = feats[f"{side}_p_ratio"]
        below_s = feats[f"{side}_wrist_below_shoulder"]
        c1 = v < TH_VERT_RATIO
        c2 = p < TH_PATH_RATIO
        c3 = below_s > -TH_RAISED_NORM
        is_support = bool(c1 and c2 and c3 and not body_too_small)
        if body_too_small:
            reason = (f"Body rise too small ({body_dy_norm:.2f} torso-lengths < "
                      f"{MIN_BODY_DY_NORM}); cannot judge reliably.")
        elif is_support:
            reason = (f"Wrist appears anchored: R_v={v:.2f} (<{TH_VERT_RATIO}), "
                      f"R_p={p:.2f} (<{TH_PATH_RATIO}), wrist below shoulder "
                      f"by {below_s:.2f} torso-lengths.")
        else:
            unmet = []
            if not c1: unmet.append(f"R_v={v:.2f} not <{TH_VERT_RATIO}")
            if not c2: unmet.append(f"R_p={p:.2f} not <{TH_PATH_RATIO}")
            if not c3: unmet.append(f"wrist raised above shoulder ({below_s:.2f})")
            reason = "Wrist appears free: " + "; ".join(unmet)
        decisions[side] = {
            "support": is_support, "reason": reason,
            "criteria": {
                "vertical_ratio_low":      bool(c1),
                "path_ratio_low":          bool(c2),
                "wrist_in_support_region": bool(c3),
            },
            "features": {
                "vertical_movement_ratio":   float(v),
                "path_ratio":                float(p),
                "wrist_below_shoulder_norm": float(below_s),
                "wrist_below_hip_norm":      float(feats.get(f"{side}_wrist_below_hip", np.nan)),
                "wrist_to_knee_norm":        float(feats.get(f"{side}_wrist_to_knee_norm", np.nan)),
                "wrist_y_std_norm":          float(feats.get(f"{side}_wrist_std_y_norm", np.nan)),
            },
        }
    overall = bool(decisions["left"]["support"] or decisions["right"]["support"])
    return overall, decisions


# ============================================================
# Visualization
# ============================================================
def _plot_skeleton(ax, L, idx, title):
    for a, b in SKELETON_EDGES:
        if a in L and b in L and idx < len(L[a]["x"]) and idx < len(L[b]["x"]):
            ax.plot([L[a]["x"][idx], L[b]["x"][idx]],
                    [L[a]["y"][idx], L[b]["y"][idx]],
                    color="#3b6cb7", linewidth=2.5, zorder=2)
    for lm_id, d in L.items():
        if idx < len(d["x"]):
            ax.scatter(d["x"][idx], d["y"][idx], color="#3b6cb7", s=22, zorder=3)
    if 15 in L:
        ax.scatter(L[15]["x"][idx], L[15]["y"][idx], color="#e74c3c", s=140,
                   zorder=4, edgecolor="black", linewidth=1.5, label="Left wrist")
    if 16 in L:
        ax.scatter(L[16]["x"][idx], L[16]["y"][idx], color="#f39c12", s=140,
                   zorder=4, edgecolor="black", linewidth=1.5, label="Right wrist")
    xmin, xmax = ax.get_xlim()
    pad_x = max(0.2 * (xmax - xmin), 0.5)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.tick_params(labelsize=8)
    ax.legend(loc="best", fontsize=8, framealpha=0.85)


def _plot_y_traj(ax, L):
    sh_y = (L[11]["y"] + L[12]["y"]) / 2
    hp_y = (L[23]["y"] + L[24]["y"]) / 2
    body_y = (sh_y + hp_y) / 2
    n = len(body_y)
    t = np.arange(n) / max(1, n - 1)
    ax.plot(t, body_y - body_y[0], label="Body center", linewidth=2.6, color="#2c3e50")
    if 15 in L:
        ax.plot(t, L[15]["y"] - L[15]["y"][0], label="Left wrist",
                linewidth=2.0, color="#e74c3c")
    if 16 in L:
        ax.plot(t, L[16]["y"] - L[16]["y"][0], label="Right wrist",
                linewidth=2.0, color="#f39c12")
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_xlabel("Time within rise (normalized)")
    ax.set_ylabel("Δy from rise start (image px, up = rising)")
    ax.set_title("Vertical trajectories: body vs wrists")
    ax.invert_yaxis(); ax.legend(loc="best", fontsize=9); ax.grid(alpha=0.3)


def _plot_features(ax, feats):
    metrics = [("v_ratio", "R_v\n(|Δy_wrist|/|Δy_body|)", TH_VERT_RATIO),
               ("p_ratio", "R_p\n(path_wrist/path_body)", TH_PATH_RATIO)]
    width = 0.35
    x = np.arange(len(metrics))
    left  = [feats.get(f"left_{m[0]}",  np.nan) for m in metrics]
    right = [feats.get(f"right_{m[0]}", np.nan) for m in metrics]
    ax.bar(x - width / 2, left,  width, color="#e74c3c", alpha=0.85, label="Left hand")
    ax.bar(x + width / 2, right, width, color="#f39c12", alpha=0.85, label="Right hand")
    for i, m in enumerate(metrics):
        th = m[2]
        ax.plot([i - width, i + width], [th, th], color="red",
                linestyle="--", linewidth=1.6)
        ax.text(i + width + 0.02, th, f" th={th}", va="center",
                fontsize=8, color="red")
    ax.set_xticks(x); ax.set_xticklabels([m[1] for m in metrics], fontsize=8)
    ax.set_ylabel("Ratio (lower = more anchored)")
    ax.set_title("Key feature ratios", fontsize=10)
    finite = [v for v in left + right if np.isfinite(v)]
    ax.set_ylim(0, max(1.2, max(finite) * 1.15) if finite else 1.2)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3, axis="y")


def _plot_decision(ax, decisions, overall):
    ax.axis("off")
    title_color = "#c0392b" if overall else "#27ae60"
    title_text = ("==>  Hand-on-external-support DETECTED"
                  if overall else
                  "==>  No hand support  (free / self-supported standing)")
    ax.text(0.5, 0.97, title_text, fontsize=13, fontweight="bold",
            color=title_color, ha="center", transform=ax.transAxes)
    y = 0.83
    for side in ("left", "right"):
        d = decisions[side]
        sym = "[SUPPORT]" if d["support"] else "[free]   "
        col = "#c0392b" if d["support"] else "#7f8c8d"
        ax.text(0.5, y, f"{sym}  {side.capitalize():>5} hand",
                fontsize=11, color=col, fontweight="bold",
                transform=ax.transAxes, family="monospace", ha="center")
        y -= 0.16
        ax.text(0.5, y, d.get("reason", ""), fontsize=9,
                transform=ax.transAxes, color="#34495e", wrap=True, ha="center")
        y -= 0.22


def _visualize(L, feats, decisions, overall, save_path, title):
    n = len(L[11]["x"])
    if n < 3:
        return
    indices = [0, n // 2, n - 1]
    titles = [f"Rise start (frame idx 0)",
              f"Rise midpoint (idx {n // 2})",
              f"Rise end (idx {n - 1})"]
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.6, 1.0, 1.0],
                  hspace=0.45, wspace=0.35)
    for i, (idx, t) in enumerate(zip(indices, titles)):
        _plot_skeleton(fig.add_subplot(gs[0, i]), L, idx, t)
    _plot_y_traj  (fig.add_subplot(gs[1, :2]), L)
    _plot_features(fig.add_subplot(gs[1, 2]),  feats)
    _plot_decision(fig.add_subplot(gs[2, :]),  decisions, overall)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Detector class
# ============================================================
class HandSupportDetector:
    """
    Hand-support detector. Stateless apart from the fps/time-range knobs;
    safe to instantiate once at app startup.
    """

    def __init__(self,
                 fps=30,
                 time_range=DEFAULT_TIME_RANGE,
                 smooth_window=5):
        self.fps = float(fps)
        self.time_range = float(time_range)
        self.smooth_window = int(smooth_window)

    def detect(self, events_df, keypoints_df,
               save_plot_dir=None,
               video_name="video"):
        """
        Decide whether each Successful event used external hand support.

        Parameters
        ----------
        events_df : pd.DataFrame
            Output of step 3.
        keypoints_df : pd.DataFrame
            Output of step 1 - the same recording's keypoints.
        save_plot_dir : str or None
            If provided, saves per-event diagnostic figures here.
        video_name : str
            Used as a filename prefix for the diagnostic figures.

        Returns
        -------
        (results_df, json_records)
            results_df is one row per Successful event. json_records contains
            the full feature/decision detail for the same events.
        """
        if events_df is None or len(events_df) == 0:
            return pd.DataFrame(), []
        if keypoints_df is None or len(keypoints_df) == 0:
            return pd.DataFrame(), []

        rows, json_rows = [], []
        for _, ev in events_df.iterrows():
            if str(ev.get("category", "")).strip().lower() != "successful":
                continue
            eid = int(ev["event_id"]) if pd.notna(ev["event_id"]) else None
            if eid is None:
                continue
            t0 = float(ev["rise_start_time"])
            t1 = float(ev["rise_end_time"])

            # Use only the early portion of the rise (patients tend to release
            # support after they're already up).
            rise_duration = t1 - t0
            t1 = t0 + self.time_range * rise_duration
            t0 = t1 - 1
            if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
                continue

            f0 = int(np.floor(t0 * self.fps))
            f1 = int(np.ceil(t1 * self.fps))
            L = _slice_landmarks(keypoints_df, f0, f1, self.smooth_window)
            if not L:
                print(f"[hand-support] no keypoints in window for event {eid}")
                continue

            feats = _compute_features(L)
            if feats is None:
                print(f"[hand-support] insufficient landmarks for event {eid}")
                continue

            overall, decisions = _decide(feats)

            if save_plot_dir:
                vis_path = os.path.join(save_plot_dir,
                                        f"{video_name}_event{eid}_decision.png")
                _visualize(L, feats, decisions, overall, vis_path,
                           f"{video_name}  |  Event {eid}  |  "
                           f"t=[{t0:.2f}, {t1:.2f}] s")

            rows.append({
                "event_id":                        eid,
                "category":                        ev.get("category", ""),
                "left_hand_support":               decisions["left"]["support"],
                "right_hand_support":              decisions["right"]["support"],
                "any_hand_support":                overall,
                "left_v_ratio":                    feats.get("left_v_ratio"),
                "left_p_ratio":                    feats.get("left_p_ratio"),
                "right_v_ratio":                   feats.get("right_v_ratio"),
                "right_p_ratio":                   feats.get("right_p_ratio"),
                "left_wrist_below_shoulder_norm":  feats.get("left_wrist_below_shoulder"),
                "right_wrist_below_shoulder_norm": feats.get("right_wrist_below_shoulder"),
                "left_wrist_to_knee_norm":         feats.get("left_wrist_to_knee_norm"),
                "right_wrist_to_knee_norm":        feats.get("right_wrist_to_knee_norm"),
                "body_dy_norm":                    feats.get("body_dy_norm"),
            })
            json_rows.append({
                "event_id":         eid,
                "category":         ev.get("category", ""),
                "rise_start_time":  t0,
                "rise_end_time":    t1,
                "overall_support":  overall,
                "decisions":        decisions,
                "features":         feats,
            })

        results_df = pd.DataFrame(rows)
        return results_df, json_rows


# ============================================================
# >>>>>>>>>>>>>>>>>>  DEBUG / DEV-ONLY  >>>>>>>>>>>>>>>>>>>>>>
# Edit the paths below and run:
#   python step4_hand_support_detection.py
# Delete this whole section for production - nothing above depends on it.
# ============================================================
if __name__ == "__main__":
    DEBUG_EVENTS_CSV = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\h_arising-from-chair\c_action-recognition\arising_event_detection.csv"
    DEBUG_KEYPOINTS_CSV = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\h_arising-from-chair\a_landmark-detection\Patient100_arising-from-chair_1\Patient100_arising-from-chair_1_keypoints.csv"
    DEBUG_OUTPUT_DIR = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\h_arising-from-chair\c_action-recognition"
    DEBUG_FPS = 30

    print(f"[DEBUG] Reading events    : {DEBUG_EVENTS_CSV}")
    events = pd.read_csv(DEBUG_EVENTS_CSV)
    print(f"[DEBUG] Reading keypoints : {DEBUG_KEYPOINTS_CSV}")
    keypoints = pd.read_csv(DEBUG_KEYPOINTS_CSV)

    detector = HandSupportDetector(fps=DEBUG_FPS)
    print(f"[DEBUG] Running detection")
    rows, json_rows = detector.detect(
        events, keypoints,
        save_plot_dir=DEBUG_OUTPUT_DIR,
        video_name="DEBUG_arising",
    )
    print(f"[DEBUG] Got {len(rows)} successful-event decisions")

    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
    csv_out  = os.path.join(DEBUG_OUTPUT_DIR, "debug_hand_support.csv")
    json_out = os.path.join(DEBUG_OUTPUT_DIR, "debug_hand_support.json")
    rows.to_csv(csv_out, index=False)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(json_rows, f, indent=2, ensure_ascii=False, default=str)
    print(f"[DEBUG] Hand-support CSV  : {csv_out}")
    print(f"[DEBUG] Hand-support JSON : {json_out}")
# ============================================================
# <<<<<<<<<<<<<<<<<<  END DEBUG / DEV-ONLY  <<<<<<<<<<<<<<<<<<
# ============================================================