"""
============================================================
特征提取 (单视频)
============================================================
后端使用方式:
    from d_motor_features import extract_features

    features = extract_features(events_df, fps=30, max_events=12)
    # features 是 18 个键的 dict, 直接喂给 step 5

`events_df` 是 step 3 的输出, 因为手指任务的事件按 segment 切分,
我们在算特征前先用 `select_representative_segment` 选出最代表性的
那一段 (第一个 >= min_events 的 segment, 否则取事件数最多的)。
============================================================
"""
import os
import numpy as np
import pandas as pd

# ============================================================
# 工具函数
# ============================================================
def safe_cv(values):
    """变异系数 = std / mean, 均值为 0 或样本不足时返回 NaN。"""
    if values is None or len(values) == 0:
        return np.nan
    mean = np.mean(values)
    if mean == 0:
        return np.nan
    return float(np.std(values, ddof=0) / mean)


def find_first_decay(amplitudes):
    """
    返回第一次出现"幅度衰减"的起始位置 (0-indexed)。
    衰减判据 (连续 3 个事件 a1, a2, a3):
      条件 A: a2 < 0.9*a1  AND  a3 < 0.9*a2
      条件 B: a2 < 0.8*a1  AND  a3 < 0.8*a1
    满足任一即视为衰减。不存在则返回 None。
    """
    for i in range(len(amplitudes) - 2):
        a1, a2, a3 = amplitudes[i], amplitudes[i + 1], amplitudes[i + 2]
        cond_a = (a2 < 0.9 * a1) and (a3 < 0.9 * a2)
        cond_b = (a2 < 0.8 * a1) and (a3 < 0.8 * a1)
        if cond_a or cond_b:
            return i + 1    #事件编号是从0开始的
    return None


# ============================================================
# 选择代表 segment
# ============================================================
def select_representative_segment(events_df, min_events=10):
    """
    手指任务的事件可能分散在多个 segment 里。这里选出"最有代表性"
    的一段:
      1. 第一个事件数 >= min_events 的 segment
      2. 否则取事件数最多的 segment

    输入空 / 单段时直接原样返回 (含全部行)。
    """
    if events_df is None or len(events_df) == 0:
        return pd.DataFrame()
    if "segment_id" not in events_df.columns:
        return events_df.copy()

    grouped = events_df.groupby("segment_id")
    segment_info = [
        {"segment_id": sid, "num_events": len(g)} for sid, g in grouped
    ]
    seg_df = pd.DataFrame(segment_info)

    valid = seg_df[seg_df["num_events"] >= min_events]
    if len(valid) > 0:
        chosen_id = valid.iloc[0]["segment_id"]
    else:
        chosen_id = seg_df.sort_values("num_events", ascending=False).iloc[0]["segment_id"]

    return events_df[events_df["segment_id"] == chosen_id].copy()


# ============================================================
# 主入口: 计算 18 个特征
# ============================================================
FEATURE_KEYS = [
    "mean_amplitude", "median_amplitude", "max_amplitude",
    "mean_duration", "median_duration", "median_movement_rate",
    "amplitude_slope", "duration_slope",
    "first_decay_position", "decay_occurred",
    "amplitude_cv", "duration_cv", "max_duration",
    "interval_cv", "max_interval",
    "num_pauses", "num_freezes", "num_events",
]


def extract_features(events_df,
                     fps=30.0,
                     max_events=12,
                     select_segment=True,
                     min_events_for_segment=10):
    """
    计算 18 个运动学特征。

    Parameters
    ----------
    events_df : pd.DataFrame
        Step 3 产物。空表也可接受 (返回全 NaN, num_events=0)。
    fps : float
    max_events : int
        特征 1-4 仅使用前 max_events 个事件; num_events 始终是全部事件数。
    select_segment : bool
        True (默认) 时先用 select_representative_segment 选段。
    min_events_for_segment : int
        select_representative_segment 的阈值。

    Returns
    -------
    dict   (JSON 可序列化, 缺失值为 Python NaN)
    """
    features = {k: np.nan for k in FEATURE_KEYS}
    features["num_events"] = 0

    if events_df is None or len(events_df) == 0:
        return features

    # 选代表段
    if select_segment:
        events_df = select_representative_segment(
            events_df, min_events=min_events_for_segment
        )
        if len(events_df) == 0:
            return features

    features["num_events"] = int(len(events_df))

    df = events_df.iloc[:max_events].copy()
    if len(df) == 0:
        return features

    amplitudes    = df["peak_amplitude"].to_numpy(dtype=float)
    durations     = df["duration_frames"].to_numpy(dtype=float) / fps
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
        intervals = (start_frames[1:] - end_frames[:-1]) / fps
        features["interval_cv"]  = safe_cv(intervals)
        features["max_interval"] = (
            float(np.max(intervals)) if len(intervals) > 0 else np.nan
        )

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
# ============================================================
if __name__ == "__main__":
    import json

    DEBUG_EVENT_CSV = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\c_hand-movements\c_event-segmentation\debug_event_events.csv"
    DEBUG_OUTPUT_JSON = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\c_hand-movements\d_rate\debug_features.json"
    DEBUG_FPS = 30.0

    print(f"[DEBUG] 读取事件: {DEBUG_EVENT_CSV}")
    try:
        events = pd.read_csv(DEBUG_EVENT_CSV)
    except pd.errors.EmptyDataError:
        events = pd.DataFrame()

    print(f"[DEBUG] 提取特征 (fps={DEBUG_FPS})")
    features = extract_features(events, fps=DEBUG_FPS)

    serialisable = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                    for k, v in features.items()}
    print("[DEBUG] 特征:")
    print(json.dumps(serialisable, indent=2, ensure_ascii=False))

    if DEBUG_OUTPUT_JSON:
        os.makedirs(os.path.dirname(DEBUG_OUTPUT_JSON) or ".", exist_ok=True)
        with open(DEBUG_OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, ensure_ascii=False, indent=2)
        print(f"[DEBUG] 特征 JSON: {DEBUG_OUTPUT_JSON}")
# ============================================================
# <<<<<<<<<<<<<<<<<<  END DEBUG / DEV-ONLY  <<<<<<<<<<<<<<<<<<
# ============================================================