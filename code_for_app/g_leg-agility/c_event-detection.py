"""
============================================================
Step 3 / 5 - Event detection on the kinematic signal
============================================================
Backend usage:
    from step3_event_detection import detect_events

    events_df = detect_events(
        kinematic_df,                # output of step 2
        fps=30,
        save_plot_path=None,         # set a path to also save the diagnostic figure
    )

`events_df` columns:
    event_index, start_frame, end_frame, duration_frames,
    peak_amplitude, rise90_frame, fall90_frame
Returns an empty DataFrame (with these columns) when no valid events.
============================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, butter, filtfilt, welch

# Force a non-interactive backend so this runs cleanly in a backend / headless
# environment. Must be set before pyplot is imported.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Helpers (unchanged from original)
# ============================================================
WIDTH_THRESHOLD = 4


def AMPD(data):
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
    min_index = np.argmin(arr_rowsum)
    max_window_length = min_index
    for k in range(1, max_window_length + 1):
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                p_data[i] += 1
    print(f"Optimal window length (L): {max_window_length}")
    return np.where(p_data >= 0.8 * max_window_length)[0]


def detect_valleys(signal):
    return AMPD(-signal)


def filter_valleys(v_abs, distance_processed, global_std, window_size_std,
                   height_threshold=0.5, variance_ratio=0.1, verbose=True):
    if len(v_abs) == 0:
        return np.array([]), np.array([])
    v_abs = np.sort(v_abs)
    valid_mask = []
    for i in range(len(v_abs)):
        idx = v_abs[i]
        left = max(0, idx - window_size_std)
        right = min(len(distance_processed), idx + window_size_std)
        local_std = np.std(distance_processed[left:right])
        condition_variance = local_std >= global_std * variance_ratio
        condition_structure = False
        if 0 < i < len(v_abs) - 1:
            left_valley = v_abs[i - 1]
            right_valley = v_abs[i + 1]
            if idx > left_valley and right_valley > idx:
                left_max = np.max(distance_processed[left_valley:idx])
                right_max = np.max(distance_processed[idx:right_valley])
                valley_value = distance_processed[idx]
                if (left_max - valley_value >= height_threshold) and \
                   (right_max - valley_value >= height_threshold):
                    condition_structure = True
                else:
                    if verbose:
                        print(f"波谷 {idx} 结构不强：左峰 {left_max:.2f}, "
                              f"右峰 {right_max:.2f}, 谷值 {valley_value:.2f}")
        valid_mask.append(condition_variance and condition_structure)
    valid_mask = np.array(valid_mask)
    return v_abs[valid_mask], v_abs[~valid_mask]


def estimate_noise_cutoff(signal, fs, ratio=0.95):
    if len(signal) < 30:
        return None
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    if np.any(np.isnan(psd)):
        return None
    mask = freqs >= 0.5
    freqs_sel = freqs[mask]
    psd_sel = psd[mask]
    if len(psd_sel) == 0:
        return None
    total_power = np.sum(psd_sel)
    if total_power <= 1e-12:
        return None
    cumulative = np.cumsum(psd_sel) / total_power
    idx = np.where(cumulative >= ratio)[0]
    if len(idx) == 0:
        return None
    return max(freqs_sel[idx[0]], 2)


def highpass_filter(data, cutoff=5, fs=30, order=4):
    data = np.asarray(data)
    if len(data) < 20:
        return np.zeros_like(data)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    padlen = 3 * max(len(a), len(b))
    if len(data) <= padlen:
        return np.zeros_like(data)
    return filtfilt(b, a, data)


def find_intersections(x, y, threshold):
    intersections = []
    for i in range(len(y) - 1):
        if (y[i] - threshold) * (y[i + 1] - threshold) < 0:
            weight = (threshold - y[i]) / (y[i + 1] - y[i])
            inter_x = x[i] + weight * (x[i + 1] - x[i])
            intersections.append(inter_x)
    return np.array(intersections)


# ============================================================
# Main entry
# ============================================================
EMPTY_EVENT_COLUMNS = [
    "event_index", "start_frame", "end_frame", "duration_frames",
    "peak_amplitude", "rise90_frame", "fall90_frame",
]


def detect_events(kinematic_df,
                  fps=30,
                  window_length=5,
                  polyorder=3,
                  window_size_std=10,
                  window_size_mean=60,
                  save_plot_path=None,
                  signal_title="Original Signal: Relative Vertical Distance"):
    """
    Run the full event-detection pipeline on a single patient's kinematic
    signal. Returns a DataFrame of detected events (possibly empty).
    """
    if kinematic_df is None or len(kinematic_df) == 0:
        return pd.DataFrame(columns=EMPTY_EVENT_COLUMNS)
    if "frame" not in kinematic_df.columns or "distance" not in kinematic_df.columns:
        raise ValueError("kinematic_df must contain 'frame' and 'distance' columns")

    frame = kinematic_df["frame"].values
    distance = kinematic_df["distance"].values.astype(float)

    # 1. S-G filter
    wl = window_length if window_length % 2 != 0 else window_length + 1
    distance_processed = savgol_filter(
        distance,
        min(wl, len(distance)),
        min(polyorder, wl - 1),
    )

    # 1.5 Quick amplitude pre-check
    signal_range = np.max(distance_processed) - np.min(distance_processed)
    if signal_range <= 3:
        print(f"信号振幅范围 {signal_range:.3f} ≤ 3，判定为无有效事件")
        if save_plot_path:
            _plot_no_event_low_range(frame, distance, distance_processed, save_plot_path,
                                     signal_title=signal_title)
        return pd.DataFrame(columns=EMPTY_EVENT_COLUMNS)

    # Build the full 4-row figure (only used if save_plot_path is set)
    fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    axes[0].plot(frame, distance, color="blue", linewidth=1)
    axes[0].set_title(f"Subplot 1: {signal_title}")

    # 2. Baseline estimation
    global_average = np.average(distance_processed)
    global_std = np.std(distance_processed)

    distance_detrended = np.full_like(distance_processed, np.nan)
    start = 0
    end = len(distance_processed)
    v_rel = detect_valleys(distance_processed)
    v_abs = start + v_rel

    # Step 0: merge nearby valleys
    merged_v_abs = []
    for idx in v_abs:
        if len(merged_v_abs) == 0:
            merged_v_abs.append(idx)
        else:
            if idx - merged_v_abs[-1] <= 5:
                print(f"波谷 {idx} 与前一个波谷 {merged_v_abs[-1]} 距离 <=5，删除 {idx}")
                continue
            merged_v_abs.append(idx)
    v_abs = np.array(merged_v_abs)

    # Filter valleys
    v_final, v_invalid = filter_valleys(v_abs, distance_processed,
                                        global_std, window_size_std)

    bx, by = [], []
    if len(v_final) > 0:
        bx = list(frame[v_final])
        by = list(distance_processed[v_final])

        # Edge handling
        if distance_processed[0] < by[0]:
            bx.insert(0, frame[0]); by.insert(0, distance_processed[0])
        else:
            bx.insert(0, frame[0]); by.insert(0, by[0])
        if distance_processed[-1] < by[-1]:
            bx.append(frame[-1]); by.append(distance_processed[-1])
        else:
            bx.append(frame[-1]); by.append(by[-1])

        baseline = np.interp(frame, bx, by)
        bx_original = bx.copy()
        by_original = by.copy()
        v_original = v_final.copy()

        signal_corrected_original = distance_processed - baseline
        v_final = sorted(v_final)
        event_median_amplitudes = []
        event_peak_indices = []
        for i in range(len(v_final) - 1):
            s = v_final[i]; e = v_final[i + 1]
            if e <= s:
                continue
            segment = signal_corrected_original[s:e + 1]
            if len(segment) < 3:
                continue
            peak_rel = np.argmax(segment)
            peak_idx = s + peak_rel
            peak_value = segment[peak_rel]
            left_valley_value = signal_corrected_original[s]
            right_valley_value = signal_corrected_original[e]
            valley_mean = (left_valley_value + right_valley_value) / 2
            amplitude = peak_value - valley_mean
            event_median_amplitudes.append(amplitude)
            event_peak_indices.append(peak_idx)
        median_amplitude = np.median(event_median_amplitudes) if event_median_amplitudes else 0

        # Add supplemental valleys
        below_mask = distance_processed < baseline
        diff_mask = np.diff(below_mask.astype(int))
        start_idxs = np.where(diff_mask == 1)[0] + 1
        end_idxs = np.where(diff_mask == -1)[0]
        if below_mask[0]:
            start_idxs = np.insert(start_idxs, 0, 0)
        if below_mask[-1]:
            end_idxs = np.append(end_idxs, len(distance_processed) - 1)

        additional_valleys_local = []
        for s_idx, e_idx in zip(start_idxs, end_idxs):
            if np.any((v_final >= s_idx) & (v_final <= e_idx)):
                continue
            local_min_idx = s_idx + np.argmin(distance_processed[s_idx:e_idx + 1])
            additional_valleys_local.append(local_min_idx)

        min_region_length = 5

        # Left edge
        if not below_mask[0]:
            left_high_mask = distance_processed[:v_final[0] + 1] > baseline[:v_final[0] + 1]
            diff_left = np.diff(left_high_mask.astype(int))
            start_idxs = np.where(diff_left == 1)[0] + 1
            end_idxs = np.where(diff_left == -1)[0]
            if left_high_mask[0]:
                start_idxs = np.insert(start_idxs, 0, 0)
            if left_high_mask[-1]:
                end_idxs = np.append(end_idxs, len(left_high_mask) - 1)
            for s_idx, e_idx in zip(start_idxs, end_idxs):
                if e_idx - s_idx + 1 < min_region_length:
                    continue
                segment = distance_processed[s_idx:e_idx + 1]
                local_min_candidates = [k for k in range(1, len(segment) - 1)
                                        if segment[k - 1] > segment[k] < segment[k + 1]]
                if not local_min_candidates:
                    continue
                best_rel = local_min_candidates[np.argmin(segment[local_min_candidates])]
                local_min_idx = s_idx + best_rel
                valley_value = distance_processed[local_min_idx]
                right_segment = distance_processed[local_min_idx:v_final[0] + 1]
                local_max_candidates = [k for k in range(1, len(right_segment) - 1)
                                        if right_segment[k - 1] < right_segment[k] > right_segment[k + 1]]
                if not local_max_candidates:
                    continue
                nearest_rel = local_max_candidates[0]
                peak_value = distance_processed[local_min_idx + nearest_rel]
                if peak_value - valley_value >= 0.25 * median_amplitude:
                    additional_valleys_local.append(local_min_idx)
                break

        # Right edge
        if not below_mask[-1]:
            right_start = v_final[-1]
            right_high_mask = distance_processed[right_start:] > baseline[right_start:]
            diff_right = np.diff(right_high_mask.astype(int))
            start_idxs = np.where(diff_right == 1)[0] + 1
            end_idxs = np.where(diff_right == -1)[0]
            if right_high_mask[0]:
                start_idxs = np.insert(start_idxs, 0, 0)
            if right_high_mask[-1]:
                end_idxs = np.append(end_idxs, len(right_high_mask) - 1)
            for s_rel, e_rel in zip(start_idxs, end_idxs):
                if e_rel - s_rel + 1 < min_region_length:
                    continue
                s_idx = right_start + s_rel
                e_idx = right_start + e_rel
                segment = distance_processed[s_idx:e_idx + 1]
                local_min_candidates = [k for k in range(1, len(segment) - 1)
                                        if segment[k - 1] > segment[k] < segment[k + 1]]
                if not local_min_candidates:
                    continue
                best_rel = local_min_candidates[np.argmin(segment[local_min_candidates])]
                local_min_idx = s_idx + best_rel
                valley_value = distance_processed[local_min_idx]
                left_segment = distance_processed[v_final[-1]:local_min_idx + 1]
                local_max_candidates = [k for k in range(1, len(left_segment) - 1)
                                        if left_segment[k - 1] < left_segment[k] > left_segment[k + 1]]
                if not local_max_candidates:
                    continue
                nearest_rel = local_max_candidates[-1]
                peak_value = distance_processed[v_final[-1] + nearest_rel]
                if peak_value - valley_value >= 0.25 * median_amplitude:
                    additional_valleys_local.append(local_min_idx)
                break

        # Recompute baseline if new valleys found
        if len(additional_valleys_local) > 0:
            v_final = np.sort(np.concatenate([v_final, additional_valleys_local]))
            bx = list(frame[v_final]); by = list(distance_processed[v_final])
            if distance_processed[0] < by[0]:
                bx.insert(0, frame[0]); by.insert(0, distance_processed[0])
            else:
                bx.insert(0, frame[0]); by.insert(0, by[0])
            if distance_processed[-1] < by[-1]:
                bx.append(frame[-1]); by.append(distance_processed[-1])
            else:
                bx.append(frame[-1]); by.append(by[-1])
            baseline = np.interp(frame, bx, by)

        distance_detrended = distance_processed - baseline

        if len(additional_valleys_local) > 0:
            additional_valleys_global = start + np.array(additional_valleys_local)
        else:
            additional_valleys_global = np.array([])
    else:
        v_original = np.array([], dtype=int)
        bx_original = []; by_original = []
        bx = []; by = []
        additional_valleys_global = np.array([], dtype=int)
        distance_detrended = np.zeros_like(distance_processed)

    # Subplot 2 plotting
    legend_added = {"valid": False, "invalid": False, "added": False,
                    "baseline_ori": False, "baseline_cor": False}
    axes[1].plot(frame, distance_processed, color="blue", linewidth=1)
    axes[1].plot(bx_original, by_original, color="limegreen", linestyle="--",
                 linewidth=1.5, zorder=2, label="Original Baseline")
    axes[1].plot(bx, by, color="green", linewidth=2.5, zorder=3,
                 label="Corrected Baseline")
    if len(v_invalid) > 0:
        axes[1].scatter(frame[v_invalid], distance_processed[v_invalid],
                        color="red", s=30, marker="o", alpha=0.6, zorder=4,
                        label="Filtered Valley")
    axes[1].scatter(frame[v_original], distance_processed[v_original],
                    color="black", s=25, marker="o", zorder=5, label="Original Valley")
    additional_valleys_global = np.array(additional_valleys_global)
    v_final = np.array(v_final)
    additional_kept = np.intersect1d(additional_valleys_global, v_final)
    if len(additional_kept) > 0:
        axes[1].scatter(frame[additional_kept], distance_processed[additional_kept],
                        color="blue", s=50, marker="D", zorder=6, label="Added Valley")
    axes[1].set_title("Subplot 2: S-G Filtered Signal & Valleys")
    axes[1].legend(loc="upper right")

    # Second amplitude check
    global_amp = np.max(distance_detrended) - np.min(distance_detrended)
    if global_amp < 2:
        print(f"全局幅值 {global_amp:.4f} < 2，判定为无有效事件")
        axes[2].plot(frame, distance_detrended, color="purple", linewidth=1.2,
                     marker="o", markersize=2)
        axes[2].set_title("Subplot 3: Detrended Signal (No Event: Amplitude < 2)")
        axes[3].set_title("Subplot 4: Event Amplitudes (None)")
        axes[3].set_ylabel("Peak Amplitude"); axes[3].set_xlabel("Frame")
        _save_or_close(fig, save_plot_path)
        return pd.DataFrame(columns=EMPTY_EVENT_COLUMNS)

    # 3. Threshold-based event detection
    cutoff = estimate_noise_cutoff(distance_detrended, fps, ratio=0.95)
    if cutoff is None or cutoff < 5.22:
        cutoff = estimate_noise_cutoff(distance_detrended, fps, ratio=0.99)
        used_ratio = 0.99
    else:
        used_ratio = 0.95
    print(f"截止频率是 {cutoff:.2f} Hz | 使用 ratio={used_ratio}"
          if cutoff is not None else f"截止频率无法计算 | 使用 ratio={used_ratio}")

    if cutoff is not None:
        hf = highpass_filter(distance_detrended, cutoff=cutoff, fs=fps)
        hf_amplitude_raw = hf.max()
    else:
        hf_amplitude_raw = 0
    if hf_amplitude_raw >= 3:
        hf_amplitude = 3
        use_mixed_threshold = True
    else:
        hf_amplitude = hf_amplitude_raw
        use_mixed_threshold = False

    event_signal_round1 = np.zeros_like(distance_processed)
    event_signal_final = np.zeros_like(distance_processed)
    event_signal_preparation = np.zeros_like(distance_processed)
    final_valid_events = []

    polar_inters = []
    for j in range(len(distance_detrended) - 1):
        if distance_detrended[j] <= hf_amplitude < distance_detrended[j + 1]:
            polar_inters.append((j, 1))
        elif distance_detrended[j] >= hf_amplitude > distance_detrended[j + 1]:
            polar_inters.append((j, -1))

    accepted_events = []
    rejected_events = []
    k = 0
    while k < len(polar_inters) - 1:
        idx_start, dir_start = polar_inters[k]
        idx_end, dir_end = polar_inters[k + 1]
        if dir_start == 1 and dir_end == -1:
            width = idx_end - idx_start
            width_threshold = (fps / cutoff - 2) if cutoff is not None else WIDTH_THRESHOLD
            if width >= width_threshold:
                accepted_events.append((idx_start, idx_end))
            else:
                rejected_events.append((idx_start, idx_end))
            k += 2
        else:
            k += 1

    # Width / height recovery
    if len(accepted_events) > 0:
        widths = [e[1] - e[0] for e in accepted_events]
        average_width = np.average(widths)
        heights = [np.max(distance_detrended[e[0]:e[1] + 1]) for e in accepted_events]
        median_height = np.median(heights)
        for idx_start, idx_end in rejected_events:
            width = idx_end - idx_start
            height = np.max(distance_detrended[idx_start:idx_end + 1])
            if width >= average_width / 2:
                accepted_events.append((idx_start, idx_end))
                print("挽回一个（宽度）")
                continue
            if height >= median_height / 2:
                accepted_events.append((idx_start, idx_end))
                print("挽回一个（高度）")
        accepted_events = sorted(set(accepted_events), key=lambda x: x[0])
        for idx_start, idx_end in accepted_events:
            event_max = np.max(distance_detrended[idx_start:idx_end + 1])
            event_signal_round1[idx_start + 1:idx_end + 1] = event_max

    # Round 2: low-peak filtering
    if len(accepted_events) > 0:
        accepted_array = np.array(sorted(accepted_events, key=lambda x: x[0]))
        event_peaks = np.array([np.max(distance_detrended[s:e + 1])
                                for s, e in accepted_array])
        A_average = np.average(event_peaks)
        low_mask = event_peaks < 0.5 * A_average
        normal_mask = ~low_mask
        final_keep_local = np.ones(len(accepted_array), dtype=bool)
        if np.sum(normal_mask) >= 2:
            normal_events = accepted_array[normal_mask]
            normal_gaps = [normal_events[i_n][0] - normal_events[i_n - 1][1]
                           for i_n in range(1, len(normal_events))]
            if normal_gaps:
                T_med = np.median(normal_gaps)
                for i_evt in range(len(accepted_array)):
                    if not low_mask[i_evt]:
                        continue
                    L_i, R_i = accepted_array[i_evt]
                    left_dist = np.inf; right_dist = np.inf
                    for (L_j, R_j) in normal_events:
                        if R_j <= L_i:
                            left_dist = L_i - R_j
                        if L_j >= R_i:
                            right_dist = L_j - R_i
                            break
                    d_i = min(left_dist, right_dist)
                    if d_i < 0.5 * T_med:
                        final_keep_local[i_evt] = False
        accepted_events_final = [tuple(accepted_array[i])
                                 for i in range(len(accepted_array))
                                 if final_keep_local[i]]
    else:
        accepted_events_final = []

    # Round 3: preparation-insufficient filter
    preparation_events = []
    if len(accepted_events_final) >= 2:
        accepted_events_final = list(accepted_events_final)
        while len(accepted_events_final) >= 2:
            event_peaks = np.array([np.max(distance_detrended[L:R + 1])
                                    for L, R in accepted_events_final])
            event_widths = np.array([R - L + 1 for L, R in accepted_events_final])
            median_width = np.median(event_widths)
            remove_count = 0
            for n in range(1, len(event_peaks)):
                threshold_peak = event_peaks[n]
                if all(event_peaks[i] < 0.333 * threshold_peak for i in range(n)):
                    remove_count = n
                    break
            width_flag = event_widths[0] > 3 * median_width
            if remove_count > 0:
                for _ in range(remove_count):
                    preparation_events.append(accepted_events_final.pop(0))
                print(f"检测到 {remove_count} 个准备不足事件，已删除")
            elif width_flag:
                preparation_events.append(accepted_events_final.pop(0))
                print("检测到宽度异常准备事件，已删除第一个事件")
            else:
                break
        for idx_start, idx_end in preparation_events:
            event_max = np.max(distance_detrended[idx_start:idx_end + 1])
            event_signal_preparation[idx_start + 1:idx_end + 1] = event_max

    for idx_start, idx_end in accepted_events_final:
        final_valid_events.append({"left": frame[idx_start], "right": frame[idx_end]})
        event_max = np.max(distance_detrended[idx_start:idx_end + 1])
        event_signal_final[idx_start + 1:idx_end + 1] = event_max

    # Subplots 3 and 4
    legend_texts = []
    if not np.all(np.isnan(distance_detrended)):
        axes[2].plot(frame, distance_detrended, color="purple", linewidth=1.2,
                     marker="o", markersize=2)
        if hf_amplitude is not None:
            axes[2].hlines(hf_amplitude, frame[0], frame[-1],
                           colors="red", linestyles="--", linewidth=1.5)
            inters = find_intersections(frame, distance_detrended, hf_amplitude)
            if len(inters) > 0:
                axes[2].scatter(inters, [hf_amplitude] * len(inters),
                                color="red", s=10, marker="o", zorder=10)
        cutoff_str = "None" if cutoff is None else f"{cutoff:.2f}"
        hf_amplitude_str = "None" if hf_amplitude is None else f"{hf_amplitude:.4f}"
        ratio_str = "None" if used_ratio is None else f"{(1 - used_ratio) * 100:.1f}%"
        if use_mixed_threshold:
            legend_texts.append(f"Cutoff={cutoff_str}Hz Amp={hf_amplitude_str}"
                                f"(Raw={hf_amplitude_raw:.2f}>3) "
                                f"High-freq Ratio={ratio_str}")
        else:
            legend_texts.append(f"Cutoff={cutoff_str}Hz Amp={hf_amplitude_str} "
                                f"High-freq Ratio={ratio_str}")
    axes[2].set_title("Subplot 3: Detrended Signal & Segment-wise Thresholds")
    axes[2].text(0.01, 0.98, "\n".join(legend_texts), transform=axes[2].transAxes,
                 fontsize=8, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    axes[3].set_title("Subplot 4: Event Amplitudes")
    axes[3].set_ylabel("Peak Amplitude"); axes[3].set_xlabel("Frame")
    axes[3].plot(frame, event_signal_round1, color="darkgreen", linewidth=1.5,
                 drawstyle="steps-post", label="Width-filtered Events")
    axes[3].fill_between(frame, 0, event_signal_final, where=event_signal_final > 0,
                         step="post", alpha=0.3, color="green",
                         label="Final Valid Events")
    axes[3].fill_between(frame, 0, event_signal_preparation,
                         where=event_signal_preparation > 0,
                         step="post", alpha=0.3, color="red",
                         label="Preparation Insufficient")
    axes[3].legend(loc="upper right")

    _save_or_close(fig, save_plot_path)

    # Build event list
    event_list = []
    event_index = 0
    i = 0
    while i < end:
        if event_signal_final[i] > 0:
            event_start = i
            amp = event_signal_final[i]
            while i < end and np.isclose(event_signal_final[i], amp):
                i += 1
            event_end = i - 1
            threshold90 = 0.9 * amp
            rise90 = next((kk for kk in range(event_start, event_end + 1)
                          if distance_detrended[kk] >= threshold90), None)
            fall90 = next((kk for kk in range(event_end, event_start - 1, -1)
                          if distance_detrended[kk] >= threshold90), None)
            event_list.append({
                "event_index": event_index,
                "start_frame": frame[event_start],
                "end_frame": frame[event_end],
                "duration_frames": event_end - event_start + 1,
                "peak_amplitude": amp,
                "rise90_frame": frame[rise90] if rise90 is not None else np.nan,
                "fall90_frame": frame[fall90] if fall90 is not None else np.nan,
            })
            event_index += 1
        else:
            i += 1
    print("处理完成。")
    return pd.DataFrame(event_list, columns=EMPTY_EVENT_COLUMNS)


# ----- internal plotting helpers -----
def _plot_no_event_low_range(frame, distance, distance_processed,
                             save_plot_path, signal_title):
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    axes[0].plot(frame, distance, color="blue", linewidth=1)
    axes[0].set_title(f"Subplot 1: {signal_title}")
    axes[1].plot(frame, distance_processed, color="blue", linewidth=1)
    axes[1].set_title("Subplot 2: S-G Filtered Signal (No Event)")
    max_val = np.max(distance_processed); min_val = np.min(distance_processed)
    axes[1].hlines(max_val, frame[0], frame[-1], colors="red", linestyles="--",
                   linewidth=1.5, label="Max Value")
    axes[1].hlines(min_val, frame[0], frame[-1], colors="green", linestyles="--",
                   linewidth=1.5, label="Min Value")
    axes[1].legend(loc="upper right")
    axes[2].plot(frame, np.zeros_like(distance_processed), color="purple")
    axes[2].set_title("Subplot 3: Event Amplitudes (No Event)")
    axes[2].set_ylabel("Peak Amplitude"); axes[2].set_xlabel("Frame")
    _save_or_close(fig, save_plot_path)


def _save_or_close(fig, save_plot_path):
    if save_plot_path:
        os.makedirs(os.path.dirname(save_plot_path) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_plot_path, dpi=300)
    plt.close(fig)


# ============================================================
# >>>>>>>>>>>>>>>>>>  DEBUG / DEV-ONLY  >>>>>>>>>>>>>>>>>>>>>>
# Edit the paths below and run:  python step3_event_detection.py
# Delete this whole section for production - nothing above depends on it.
# ============================================================
if __name__ == "__main__":
    DEBUG_KINEMATIC_CSV = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\leg-agility\b_kinematic-signal\debug_kinematic_signal_norm.csv"
    DEBUG_OUTPUT_PLOT = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\leg-agility\c_event-detection\debug_events.png"
    DEBUG_OUTPUT_CSV = DEBUG_OUTPUT_PLOT.replace(".png", ".csv")
    DEBUG_FPS = 30

    print(f"[DEBUG] Reading kinematic CSV: {DEBUG_KINEMATIC_CSV}")
    kinematic = pd.read_csv(DEBUG_KINEMATIC_CSV)
    print(f"[DEBUG] Detecting events (fps={DEBUG_FPS})")
    events = detect_events(kinematic, fps=DEBUG_FPS, save_plot_path=DEBUG_OUTPUT_PLOT)
    print(f"[DEBUG] Got {len(events)} events")

    os.makedirs(os.path.dirname(DEBUG_OUTPUT_CSV) or ".", exist_ok=True)
    events.to_csv(DEBUG_OUTPUT_CSV, index=False)
    print(f"[DEBUG] Events CSV : {DEBUG_OUTPUT_CSV}")
    print(f"[DEBUG] Plot       : {DEBUG_OUTPUT_PLOT}")
# ============================================================
# <<<<<<<<<<<<<<<<<<  END DEBUG / DEV-ONLY  <<<<<<<<<<<<<<<<<<
# ============================================================