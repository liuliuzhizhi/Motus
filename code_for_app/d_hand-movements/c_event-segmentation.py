"""
============================================================
事件检测 (单视频)
============================================================
后端使用方式:
    from tapping_step3_event_detection import (
        detect_finger_tapping_events, FingerTappingConfig,
    )

    cfg = FingerTappingConfig(fps=30)
    events_df = detect_finger_tapping_events(signal_df, cfg)

输入 signal_df 列: frame, is_valid, area (Step 2 产物)
输出 events_df 列:
    segment_id, segment_start, segment_end,
    event_index_in_segment, event_global_index,
    start_frame, end_frame, duration_frames,
    peak_amplitude, rise90_frame, fall90_frame
============================================================
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, butter, filtfilt, welch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 配置 (所有可调参数集中在这里)
# ============================================================
@dataclass
class FingerTappingConfig:
    """事件检测的所有可调参数。"""
    # ----- 采样 / 滤波 -----
    fps: float = 30.0
    sg_window_length: int = 5
    sg_polyorder: int = 3

    # ----- 波谷筛选 -----
    valley_local_std_window: int = 10
    valley_local_std_ratio: float = 0.1

    # ----- 噪声截止频率 -----
    cutoff_primary_ratio: float = 0.95
    cutoff_fallback_ratio: float = 0.99
    cutoff_min_for_primary: float = 5.83

    # ----- 第一轮: 宽度筛选 -----
    width_threshold_default: int = 4
    width_recovery_ratio: float = 0.5
    height_recovery_ratio: float = 0.5

    # ----- 第二轮: 矮峰过滤 -----
    low_peak_ratio: float = 0.5
    low_peak_distance_ratio: float = 0.5

    # ----- 第三轮: 准备不足 -----
    enable_preparation_filter: bool = True
    prep_peak_ratio: float = 0.333
    prep_width_ratio: float = 3.0

    # ----- 第四轮: 高度+宽度双小 -----
    enable_outlier_filter: bool = True
    outlier_height_ratio: float = 0.4
    outlier_width_ratio: float = 0.3

    # ----- 输出 -----
    save_plot_path: Optional[str] = None
    plot_signal_title: str = "Hand Movements"


# ============================================================
# 第 1 步: 信号预处理
# ============================================================
def preprocess_signal(distance, is_valid, cfg: FingerTappingConfig):
    """插值 + 分段 S-G 滤波。"""
    filled = distance.copy()
    valid_mask = (is_valid == 1)
    if np.any(valid_mask):
        filled[valid_mask] = (
            pd.Series(filled[valid_mask]).interpolate(method="linear").values
        )

    smoothed = filled.copy()
    n = len(filled)
    wl = cfg.sg_window_length if cfg.sg_window_length % 2 != 0 else cfg.sg_window_length + 1
    i = 0
    while i < n:
        if is_valid[i] == 1:
            start = i
            while i < n and is_valid[i] == 1:
                i += 1
            seg = filled[start:i]
            if len(seg) > cfg.sg_window_length:
                smoothed[start:i] = savgol_filter(
                    seg, min(wl, len(seg)),
                    min(cfg.sg_polyorder, wl - 1),
                )
        else:
            i += 1
    return smoothed


def compute_global_stats(distance_sg, is_valid):
    """计算只取有效帧的全局均值与标准差。"""
    valid_data = distance_sg[is_valid == 1]
    return float(np.average(valid_data)), float(np.std(valid_data))


def split_valid_segments(is_valid):
    """把信号按 is_valid==1 切成若干 [start, end) 段。"""
    segments = []
    n = len(is_valid)
    i = 0
    while i < n:
        if is_valid[i] == 1:
            start = i
            while i < n and is_valid[i] == 1:
                i += 1
            segments.append((start, i))
        else:
            i += 1
    return segments


# ============================================================
# 波谷检测 + 筛选
# ============================================================
def AMPD(data):
    """Automatic Multi-scale Peak Detection. 返回峰索引。"""
    p_data = np.zeros_like(data, dtype=np.int32)
    count = data.shape[0]
    arr_rowsum = []
    for k in range(1, count // 2 + 1):
        row_sum = 0
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                row_sum -= 1
        arr_rowsum.append(row_sum)
    if not arr_rowsum:
        return np.array([])
    max_window_length = int(np.argmin(arr_rowsum))
    for k in range(1, max_window_length + 1):
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                p_data[i] += 1
    print(f"[AMPD] Optimal window length (L): {max_window_length}")
    return np.where(p_data >= 0.8 * max_window_length)[0]


def detect_segment_valleys(distance_sg, seg_start, seg_end,
                           global_average, global_std,
                           cfg: FingerTappingConfig):
    """单段内检测波谷, 按 (低于全局均值 + 局部方差足够大) 筛选。"""
    seg = distance_sg[seg_start:seg_end]
    v_rel = AMPD(-seg)
    v_abs = seg_start + v_rel

    valid_mask = []
    for idx in v_abs:
        left = max(0, idx - cfg.valley_local_std_window)
        right = min(len(distance_sg), idx + cfg.valley_local_std_window)
        local_std = float(np.std(distance_sg[left:right]))
        cond_below_avg = distance_sg[idx] < global_average
        cond_local_std = local_std >= global_std * cfg.valley_local_std_ratio
        valid_mask.append(cond_below_avg and cond_local_std)
    valid_mask = np.array(valid_mask, dtype=bool)
    return v_abs[valid_mask], v_abs[~valid_mask]


# ============================================================
# 基线构造与补充波谷
# ============================================================
def build_baseline(distance_sg, frame, valleys, seg_start, seg_end):
    """波谷折线基线, 含端点边缘逻辑。"""
    seg_f = frame[seg_start:seg_end]
    seg_d = distance_sg[seg_start:seg_end]
    bx = list(frame[valleys])
    by = list(distance_sg[valleys])
    if seg_d[0] < by[0]:
        bx.insert(0, seg_f[0]); by.insert(0, seg_d[0])
    else:
        bx.insert(0, seg_f[0]); by.insert(0, by[0])
    if seg_d[-1] < by[-1]:
        bx.append(seg_f[-1]); by.append(seg_d[-1])
    else:
        bx.append(seg_f[-1]); by.append(by[-1])
    baseline = np.interp(seg_f, bx, by)
    return baseline, bx, by


def supplement_valleys_below_baseline(seg_d, baseline, v_final_rel):
    """低于基线但没有已知波谷的连续段, 在最低点补一个波谷。"""
    below_mask = seg_d < baseline
    diff_mask = np.diff(below_mask.astype(int))
    starts = np.where(diff_mask == 1)[0] + 1
    ends = np.where(diff_mask == -1)[0]
    if below_mask[0]:
        starts = np.insert(starts, 0, 0)
    if below_mask[-1]:
        ends = np.append(ends, len(seg_d) - 1)
    new_valleys = []
    for s, e in zip(starts, ends):
        if np.any((v_final_rel >= s) & (v_final_rel <= e)):
            continue
        new_valleys.append(s + int(np.argmin(seg_d[s:e + 1])))
    return new_valleys


# ============================================================
# 噪声截止频率 + 高通阈值
# ============================================================
def estimate_noise_cutoff(signal, fs, ratio=0.95):
    """Welch 估计 ratio 累计能量对应的最低频率, 不小于 2 Hz。"""
    if len(signal) < 30:
        return None
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    if np.any(np.isnan(psd)):
        return None
    mask = freqs >= 0.5
    freqs_sel, psd_sel = freqs[mask], psd[mask]
    if len(psd_sel) == 0:
        return None
    total = float(np.sum(psd_sel))
    if total <= 1e-12:
        return None
    cumulative = np.cumsum(psd_sel) / total
    idx = np.where(cumulative >= ratio)[0]
    if len(idx) == 0:
        return None
    return float(max(freqs_sel[idx[0]], 2))


def highpass_filter(data, cutoff, fs, order=4):
    """Butterworth 高通; 数据太短返回全零。"""
    data = np.asarray(data)
    if len(data) < 20:
        return np.zeros_like(data)
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff / nyquist, btype="high", analog=False)
    padlen = 3 * max(len(a), len(b))
    if len(data) <= padlen:
        return np.zeros_like(data)
    return filtfilt(b, a, data)


def estimate_threshold_from_noise(detrended_seg, cfg, ratio):
    """估计阈值, 返回 (cutoff, hf_seg)。"""
    cutoff = estimate_noise_cutoff(detrended_seg, cfg.fps, ratio=ratio)
    if cutoff is not None:
        hf_seg = highpass_filter(detrended_seg, cutoff=cutoff, fs=cfg.fps)
    else:
        hf_seg = np.zeros_like(detrended_seg)
    return cutoff, hf_seg


# ============================================================
# 单段处理结果
# ============================================================
@dataclass
class SegmentResult:
    """单个有效段处理后的中间结果。"""
    start: int
    end: int
    v_original: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    v_invalid:  np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    v_added:    np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    bx_original: list = field(default_factory=list)
    by_original: list = field(default_factory=list)
    bx_corrected: list = field(default_factory=list)
    by_corrected: list = field(default_factory=list)
    detrended: np.ndarray = field(default_factory=lambda: np.array([]))
    hf_seg:    np.ndarray = field(default_factory=lambda: np.array([]))
    cutoff: Optional[float] = None


def process_one_segment(distance_sg, frame, seg_start, seg_end,
                        global_average, global_std, cfg, cutoff_ratio):
    """一个有效段: 检测波谷 -> 基线 -> 补谷 -> 重建基线 -> 估阈值。"""
    res = SegmentResult(start=seg_start, end=seg_end)
    seg_d = distance_sg[seg_start:seg_end]

    v_final, v_invalid = detect_segment_valleys(
        distance_sg, seg_start, seg_end, global_average, global_std, cfg
    )
    res.v_original = v_final.copy()
    res.v_invalid = v_invalid

    if len(v_final) == 0:
        res.detrended = np.zeros_like(seg_d)
        res.hf_seg    = np.zeros_like(seg_d)
        return res

    baseline, bx, by = build_baseline(
        distance_sg, frame, v_final, seg_start, seg_end
    )
    res.bx_original = bx.copy()
    res.by_original = by.copy()

    v_final_rel = v_final - seg_start
    added_rel = supplement_valleys_below_baseline(seg_d, baseline, v_final_rel)
    if added_rel:
        added_global = seg_start + np.array(added_rel)
        v_final = np.sort(np.concatenate([v_final, added_global]))
        res.v_added = added_global
        baseline, bx, by = build_baseline(
            distance_sg, frame, v_final, seg_start, seg_end
        )

    res.bx_corrected = bx
    res.by_corrected = by
    res.detrended = seg_d - baseline

    cutoff, hf_seg = estimate_threshold_from_noise(res.detrended, cfg, cutoff_ratio)
    res.cutoff = cutoff
    res.hf_seg = hf_seg
    return res


# ============================================================
# 事件检测: 阈值穿越 + 四轮过滤
# ============================================================
def find_polar_intersections(signal, threshold):
    """扫描穿越阈值位置, 返回 [(idx, dir), ...] dir=+1 上穿/-1 下穿。"""
    pol = []
    for j in range(len(signal) - 1):
        if signal[j] <= threshold < signal[j + 1]:
            pol.append((j, 1))
        elif signal[j] >= threshold > signal[j + 1]:
            pol.append((j, -1))
    return pol


def round1_width_filter(polar_inters, seg_det, cutoff, cfg):
    """配对上下穿点 -> 按宽度筛选 -> 宽度+高度补偿。"""
    accepted, rejected = [], []
    width_threshold = (cfg.fps / cutoff - 2) if cutoff is not None else cfg.width_threshold_default

    k = 0
    while k < len(polar_inters) - 1:
        idx_s, dir_s = polar_inters[k]
        idx_e, dir_e = polar_inters[k + 1]
        if dir_s == 1 and dir_e == -1:
            width = idx_e - idx_s
            if width >= width_threshold:
                accepted.append((idx_s, idx_e))
            else:
                rejected.append((idx_s, idx_e))
            k += 2
        else:
            k += 1

    if not accepted:
        return []

    avg_width = float(np.average([e[1] - e[0] for e in accepted]))
    median_height = float(np.median([
        np.max(seg_det[s:e + 1]) for s, e in accepted
    ]))
    for idx_s, idx_e in rejected:
        width = idx_e - idx_s
        height = float(np.max(seg_det[idx_s:idx_e + 1]))
        if width >= avg_width * cfg.width_recovery_ratio:
            accepted.append((idx_s, idx_e))
            print("[round1] 挽回 (宽度)")
            continue
        if height >= median_height * cfg.height_recovery_ratio:
            accepted.append((idx_s, idx_e))
            print("[round1] 挽回 (高度)")

    return sorted(set(accepted), key=lambda x: x[0])


def round2_low_peak_filter(accepted, seg_det, cfg):
    """
    第二轮 (级联两层):
      1. 全局矮峰: 峰值 < low_peak_ratio × 平均峰; 离正常峰过近 -> 删
      2. 局部矮峰 (基于存活峰): 峰值 < 0.6 × 相邻两侧存活峰;
         (过近 或 过窄) -> 删
    """
    if not accepted:
        return []
    arr = np.array(sorted(accepted, key=lambda x: x[0]))
    n = len(arr)
    peaks = np.array([np.max(seg_det[s:e + 1]) for s, e in arr])
    widths = np.array([e - s + 1 for s, e in arr])
    avg_peak = float(np.average(peaks))
    keep = np.ones(n, dtype=bool)

    # 1. 全局矮峰
    low_global = peaks < cfg.low_peak_ratio * avg_peak
    normal_mask = ~low_global
    if np.sum(normal_mask) >= 2:
        normal = arr[normal_mask]
        gaps = [normal[i][0] - normal[i - 1][1] for i in range(1, len(normal))]
        if gaps:
            T_med = float(np.median(gaps))
            for i in range(n):
                if not low_global[i]:
                    continue
                L_i, R_i = arr[i]
                left_d = right_d = np.inf
                for L_j, R_j in normal:
                    if R_j <= L_i:
                        left_d = L_i - R_j
                    if L_j >= R_i:
                        right_d = L_j - R_i
                        break
                if min(left_d, right_d) < cfg.low_peak_distance_ratio * T_med:
                    keep[i] = False

    # 2. 局部矮峰
    valid_indices = np.where(keep)[0]
    if len(valid_indices) >= 3:
        gaps_valid = [
            arr[valid_indices[i]][0] - arr[valid_indices[i - 1]][1]
            for i in range(1, len(valid_indices))
        ]
        T_med_valid = float(np.median(gaps_valid)) if gaps_valid else 1.0
        # 计算中位宽度
        valid_indices_round1 = np.where(keep)[0]
        if len(valid_indices_round1) == 0:
            return []
        widths_round1 = widths[valid_indices_round1]
        width_med = float(np.median(widths_round1)) if len(widths_round1) > 0 else 1.0

        for k in range(1, len(valid_indices) - 1):
            i = valid_indices[k]
            left_i = valid_indices[k - 1]
            right_i = valid_indices[k + 1]
            is_local_low = (
                peaks[i] < 0.6 * peaks[left_i] and
                peaks[i] < 0.6 * peaks[right_i]
            )
            if not is_local_low:
                continue
            left_gap  = arr[i][0] - arr[left_i][1]
            right_gap = arr[right_i][0] - arr[i][1]
            too_close = (
                left_gap  < cfg.low_peak_distance_ratio * T_med_valid or
                right_gap < cfg.low_peak_distance_ratio * T_med_valid
            )
            too_narrow = widths[i] < 0.3 * width_med
            if too_close or too_narrow:
                keep[i] = False

    return [tuple(arr[i]) for i in range(n) if keep[i]]


def round3_preparation_filter(accepted, seg_det, cfg):
    """删除开头的"准备不足"事件, 返回 (保留, 删除)。"""
    accepted = list(accepted)
    removed = []
    if not cfg.enable_preparation_filter:
        return accepted, removed
    while len(accepted) >= 2:
        peaks = np.array([np.max(seg_det[L:R + 1]) for L, R in accepted])
        widths = np.array([R - L + 1 for L, R in accepted])
        median_w = float(np.median(widths))
        rm_count = 0
        for n in range(1, len(peaks)):
            thresh = peaks[n]
            if all(peaks[i] < cfg.prep_peak_ratio * thresh for i in range(n)):
                rm_count = n
                break
        width_flag = widths[0] > cfg.prep_width_ratio * median_w
        if rm_count > 0:
            for _ in range(rm_count):
                removed.append(accepted.pop(0))
            print(f"[round3] 删除 {rm_count} 个准备不足事件")
        elif width_flag:
            removed.append(accepted.pop(0))
            print("[round3] 删除首个宽度异常事件")
        else:
            break
    return accepted, removed


def round4_outlier_filter(accepted, seg_det, cfg):
    """删除"高度和宽度都显著小于其他事件"的离群小事件。"""
    if not cfg.enable_outlier_filter or len(accepted) < 3:
        return list(accepted), []
    arr = sorted(accepted, key=lambda x: x[0])
    heights = np.array([np.max(seg_det[s:e + 1]) for s, e in arr])
    widths  = np.array([e - s + 1 for s, e in arr])
    median_h = float(np.median(heights))
    median_w = float(np.median(widths))
    h_thresh = cfg.outlier_height_ratio * median_h
    w_thresh = cfg.outlier_width_ratio  * median_w
    kept, removed = [], []
    for i, ev in enumerate(arr):
        if heights[i] < h_thresh and widths[i] < w_thresh:
            removed.append(ev)
            print(f"[round4] 删除离群小事件 idx={i} "
                  f"(h={heights[i]:.3f}<{h_thresh:.3f}, "
                  f"w={widths[i]}<{w_thresh:.1f})")
        else:
            kept.append(ev)
    return kept, removed


# ============================================================
# 事件抽取
# ============================================================
def extract_events(accepted_per_segment, segment_results,
                   distance_detrended, frame):
    """合并各段事件, 计算 90% 上升/下降帧, 输出标准 DataFrame。"""
    rows = []
    event_global = 0
    for seg_id, (events, seg) in enumerate(zip(accepted_per_segment, segment_results)):
        for in_seg_id, (idx_s_rel, idx_e_rel) in enumerate(events):
            idx_s = seg.start + idx_s_rel
            idx_e = seg.start + idx_e_rel
            amp = float(np.max(distance_detrended[idx_s:idx_e + 1]))
            thr90 = 0.9 * amp
            rise90 = next((k for k in range(idx_s, idx_e + 1)
                           if distance_detrended[k] >= thr90), None)
            fall90 = next((k for k in range(idx_e, idx_s - 1, -1)
                           if distance_detrended[k] >= thr90), None)
            rows.append({
                "segment_id": seg_id,
                "segment_start": frame[seg.start],
                "segment_end":   frame[seg.end - 1],
                "event_index_in_segment": in_seg_id,
                "event_global_index": event_global,
                "start_frame": frame[idx_s],
                "end_frame":   frame[idx_e],
                "duration_frames": idx_e - idx_s + 1,
                "peak_amplitude":  amp,
                "rise90_frame": frame[rise90] if rise90 is not None else np.nan,
                "fall90_frame": frame[fall90] if fall90 is not None else np.nan,
            })
            event_global += 1
    return pd.DataFrame(rows)


# ============================================================
# 绘图
# ============================================================
def _draw_invalid_regions(ax, frame, is_valid):
    n = len(is_valid); i = 0
    while i < n:
        if is_valid[i] == 0:
            start = i
            while i < n and is_valid[i] == 0:
                i += 1
            ax.axvspan(frame[start], frame[i - 1], color="red", alpha=0.15, zorder=0)
        else:
            i += 1


def _find_intersections(x, y, threshold):
    out = []
    for i in range(len(y) - 1):
        if (y[i] - threshold) * (y[i + 1] - threshold) < 0:
            w = (threshold - y[i]) / (y[i + 1] - y[i])
            out.append(x[i] + w * (x[i + 1] - x[i]))
    return np.array(out)


def plot_diagnostic(frame, distance, distance_sg, is_valid,
                    segment_results, distance_detrended,
                    hf_amplitude, used_ratio, max_cutoff,
                    event_signal_round1, event_signal_final,
                    event_signal_prep, event_signal_outlier,
                    title, save_path):
    """画 4 子图诊断图。"""
    global_average = float(np.average(distance_sg[is_valid == 1]))
    fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    for ax in axes:
        _draw_invalid_regions(ax, frame, is_valid)

    axes[0].plot(frame, distance, color="blue", linewidth=1)
    axes[0].set_title(f"Subplot 1: Original Signal ({title})")

    legend = {"valid": False, "invalid": False, "added": False,
              "baseline_ori": False, "baseline_cor": False}
    for s in segment_results:
        seg_f = frame[s.start:s.end]; seg_d = distance_sg[s.start:s.end]
        axes[1].plot(seg_f, seg_d, color="blue", linewidth=1)
        if s.bx_original:
            axes[1].plot(s.bx_original, s.by_original, color="limegreen",
                         linestyle="--", linewidth=1.5, zorder=2,
                         label="Original Baseline" if not legend["baseline_ori"] else "")
            legend["baseline_ori"] = True
        if s.bx_corrected:
            axes[1].plot(s.bx_corrected, s.by_corrected, color="green",
                         linewidth=2.5, zorder=3,
                         label="Corrected Baseline" if not legend["baseline_cor"] else "")
            legend["baseline_cor"] = True
        if len(s.v_invalid) > 0:
            axes[1].scatter(frame[s.v_invalid], distance_sg[s.v_invalid],
                            color="red", s=30, marker="o", alpha=0.6, zorder=4,
                            label="Filtered Valley" if not legend["invalid"] else "")
            legend["invalid"] = True
        if len(s.v_original) > 0:
            axes[1].scatter(frame[s.v_original], distance_sg[s.v_original],
                            color="black", s=25, marker="o", zorder=5,
                            label="Original Valley" if not legend["valid"] else "")
            legend["valid"] = True
        if len(s.v_added) > 0:
            axes[1].scatter(frame[s.v_added], distance_sg[s.v_added],
                            color="blue", s=50, marker="D", zorder=6,
                            label="Added Valley" if not legend["added"] else "")
            legend["added"] = True
    axes[1].axhline(global_average, color="orange", linestyle="--",
                    alpha=0.7, label="Average")
    axes[1].set_title("Subplot 2: S-G Filtered Signal & Valleys")
    axes[1].legend(loc="upper right")

    for s in segment_results:
        seg_f = frame[s.start:s.end]
        seg_det = distance_detrended[s.start:s.end]
        if not np.all(np.isnan(seg_det)):
            axes[2].plot(seg_f, seg_det, color="purple", linewidth=1.2,
                         marker="o", markersize=2)
            inters = _find_intersections(seg_f, seg_det, hf_amplitude)
            if len(inters) > 0:
                axes[2].scatter(inters, [hf_amplitude] * len(inters),
                                color="red", s=10, marker="o", zorder=10)
    hf_ratio = (1 - used_ratio) * 100
    axes[2].axhline(hf_amplitude, color="red", linestyle="--",
                    label=(f"Threshold={hf_amplitude:.4f} "
                           f"(Cutoff={max_cutoff:.2f}Hz, "
                           f"High-freq Ratio={hf_ratio:.1f}%)"))
    axes[2].set_title("Subplot 3: Detrended Signal & Threshold Intersections")
    axes[2].legend(loc="upper right")

    axes[3].set_title("Subplot 4: Event Amplitudes")
    axes[3].set_ylabel("Peak Amplitude"); axes[3].set_xlabel("Frame")
    axes[3].plot(frame, event_signal_round1, color="darkgreen", linewidth=1.5,
                 drawstyle="steps-post", label="Width-filtered Events")
    axes[3].fill_between(frame, 0, event_signal_final,
                         where=event_signal_final > 0, step="post",
                         alpha=0.3, color="green", label="Final Valid Events")
    axes[3].fill_between(frame, 0, event_signal_prep,
                         where=event_signal_prep > 0, step="post",
                         alpha=0.3, color="red", label="Preparation Insufficient")
    axes[3].fill_between(frame, 0, event_signal_outlier,
                         where=event_signal_outlier > 0, step="post",
                         alpha=0.3, color="orange", label="Outlier (small h&w)")
    axes[3].legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================
# 主入口
# ============================================================
def detect_finger_tapping_events(signal_df: pd.DataFrame,
                                 cfg: Optional[FingerTappingConfig] = None) -> pd.DataFrame:
    """
    事件检测主流程。

    Parameters
    ----------
    signal_df : pd.DataFrame
        Step 2 产物。需包含列: frame, is_valid, polygon_area_0_1_4_8_12_16_20

    Returns
    -------
    pd.DataFrame, 列见模块文档
    """
    cfg = cfg or FingerTappingConfig()

    if signal_df is None or len(signal_df) == 0:
        return pd.DataFrame()
    for c in ("frame", "is_valid", "polygon_area_0_1_4_8_12_16_20"):
        if c not in signal_df.columns:
            raise ValueError(f"signal_df 缺列: {c}")

    frame    = signal_df["frame"].values
    is_valid = signal_df["is_valid"].values
    distance = signal_df["polygon_area_0_1_4_8_12_16_20"].values.astype(float)
    distance_sg = preprocess_signal(distance, is_valid, cfg)

    global_average, global_std = compute_global_stats(distance_sg, is_valid)
    segments = split_valid_segments(is_valid)
    if not segments:
        print("[finger-tapping] 没有任何有效段")
        return pd.DataFrame()

    # 单段处理 (含截止频率重做兜底)
    used_ratio = cfg.cutoff_primary_ratio
    segment_results, max_cutoff = _run_all_segments(
        distance_sg, frame, segments, global_average, global_std,
        cfg, cutoff_ratio=used_ratio,
    )
    if max_cutoff <= cfg.cutoff_min_for_primary:
        print(f"[cutoff] max={max_cutoff:.2f} <= "
              f"{cfg.cutoff_min_for_primary}, 改用 ratio="
              f"{cfg.cutoff_fallback_ratio} 重做")
        used_ratio = cfg.cutoff_fallback_ratio
        segment_results, max_cutoff = _run_all_segments(
            distance_sg, frame, segments, global_average, global_std,
            cfg, cutoff_ratio=used_ratio,
        )

    # 全局阈值 + 各段事件检测 + 四轮过滤
    distance_detrended = np.full_like(distance_sg, np.nan, dtype=float)
    high_freq_all = []
    for s in segment_results:
        distance_detrended[s.start:s.end] = s.detrended
        high_freq_all.extend(s.hf_seg)
    hf_amplitude = float(np.percentile(high_freq_all, 100)) if high_freq_all else 0.0

    accepted_per_segment, round1_per_segment = [], []
    preparation_per_segment, outlier_per_segment = [], []
    for s in segment_results:
        if np.all(s.detrended == 0):
            accepted_per_segment.append([])
            round1_per_segment.append([])
            preparation_per_segment.append([])
            outlier_per_segment.append([])
            continue
        polar = find_polar_intersections(s.detrended, hf_amplitude)
        r1 = round1_width_filter(polar, s.detrended, s.cutoff, cfg)
        r2 = round2_low_peak_filter(r1, s.detrended, cfg)
        r3, prep_removed    = round3_preparation_filter(r2, s.detrended, cfg)
        r4, outlier_removed = round4_outlier_filter(r3, s.detrended, cfg)
        accepted_per_segment.append(r4)
        round1_per_segment.append(r1)
        preparation_per_segment.append(prep_removed)
        outlier_per_segment.append(outlier_removed)

    # 画图 (可选)
    if cfg.save_plot_path:
        ev_round1   = _build_event_signal(round1_per_segment,      segment_results, distance_detrended, len(distance_sg))
        ev_final    = _build_event_signal(accepted_per_segment,    segment_results, distance_detrended, len(distance_sg))
        ev_prep     = _build_event_signal(preparation_per_segment, segment_results, distance_detrended, len(distance_sg))
        ev_outlier  = _build_event_signal(outlier_per_segment,     segment_results, distance_detrended, len(distance_sg))
        plot_diagnostic(
            frame, distance, distance_sg, is_valid,
            segment_results, distance_detrended,
            hf_amplitude=hf_amplitude, used_ratio=used_ratio, max_cutoff=max_cutoff,
            event_signal_round1=ev_round1, event_signal_final=ev_final,
            event_signal_prep=ev_prep, event_signal_outlier=ev_outlier,
            title=cfg.plot_signal_title, save_path=cfg.save_plot_path,
        )

    # 抽取事件
    event_df = extract_events(
        accepted_per_segment, segment_results, distance_detrended, frame
    )
    print(f"[done] {len(event_df)} 个事件 | 全局阈值 {hf_amplitude:.4f} | "
          f"max_cutoff {max_cutoff:.2f}Hz | ratio {used_ratio}")
    return event_df


def _run_all_segments(distance_sg, frame, segments, global_average, global_std,
                      cfg, cutoff_ratio):
    """对所有段按给定 ratio 跑一遍, 返回 (结果列表, 最大 cutoff)。"""
    results = []
    max_cutoff = 0.0
    for seg_start, seg_end in segments:
        res = process_one_segment(
            distance_sg, frame, seg_start, seg_end,
            global_average, global_std, cfg, cutoff_ratio,
        )
        if res.cutoff is not None and res.cutoff > max_cutoff:
            max_cutoff = res.cutoff
        results.append(res)
    return results, max_cutoff


def _build_event_signal(events_per_segment, segment_results,
                        distance_detrended, total_length):
    """事件区间列表 -> 1D 数组, 用于阶梯图。"""
    out = np.zeros(total_length, dtype=float)
    for events, seg in zip(events_per_segment, segment_results):
        for idx_s_rel, idx_e_rel in events:
            idx_s = seg.start + idx_s_rel
            idx_e = seg.start + idx_e_rel
            amp = float(np.max(distance_detrended[idx_s:idx_e + 1]))
            out[idx_s + 1:idx_e + 1] = amp
    return out


# ============================================================
# >>>>>>>>>>>>>>>>>>  DEBUG / DEV-ONLY  >>>>>>>>>>>>>>>>>>>>>>
# ============================================================
if __name__ == "__main__":
    DEBUG_SIGNAL_CSV = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\c_hand-movements\b_kinematic-signal\debug_polygon_area.csv"
    DEBUG_OUTPUT_PLOT = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\c_hand-movements\c_event-segmentation\debug_event.png"
    DEBUG_OUTPUT_CSV = DEBUG_OUTPUT_PLOT.replace(".png", "_events.csv")

    print(f"[DEBUG] 读取信号: {DEBUG_SIGNAL_CSV}")
    sig = pd.read_csv(DEBUG_SIGNAL_CSV)

    cfg = FingerTappingConfig(
        fps=30,
        save_plot_path=DEBUG_OUTPUT_PLOT,
        plot_signal_title="DEBUG hand-movements",
    )
    print("[DEBUG] 开始检测")
    events = detect_finger_tapping_events(sig, cfg)
    print(f"[DEBUG] 共 {len(events)} 个事件")

    os.makedirs(os.path.dirname(DEBUG_OUTPUT_CSV) or ".", exist_ok=True)
    events.to_csv(DEBUG_OUTPUT_CSV, index=False)
    print(f"[DEBUG] 事件 CSV: {DEBUG_OUTPUT_CSV}")
    print(f"[DEBUG] 诊断图  : {DEBUG_OUTPUT_PLOT}")
# ============================================================
# <<<<<<<<<<<<<<<<<<  END DEBUG / DEV-ONLY  <<<<<<<<<<<<<<<<<<
# ============================================================