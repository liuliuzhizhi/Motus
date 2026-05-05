"""
============================================================
手指轻叩 Step 5 / 5 - 规则打分 (单视频)
============================================================
后端使用方式:
    from tapping_step5_rule_scoring import score_finger_tapping

    result = score_finger_tapping(features, normal_rate=2.514)

`features` 是 step 4 返回的 dict。
`result` 是 JSON 可序列化 dict, 包含:
    final_score, speed_score, speed_level, speed_reason,
    pause_score, decay_score, base_score,
    special_rule, rule_triggered, score_explanation, status

`status`:
    "ok"                  正常打分
    "error: ..."          输入有问题, final_score=None, 不要使用
============================================================
"""
import os
import math


REQUIRED_FEATURE_KEYS = (
    "median_movement_rate",
    "num_pauses", "num_freezes",
    "decay_occurred", "first_decay_position",
    "num_events", "max_amplitude",
)


def _is_missing(v):
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return False


def score_finger_tapping(features, normal_rate=2.514):
    """
    手指轻叩任务的规则打分 (基于 MDS-UPDRS 3.4)。

    Parameters
    ----------
    features : dict
        Step 4 输出。需包含: median_movement_rate, num_pauses,
        num_freezes, decay_occurred, first_decay_position,
        num_events, max_amplitude
    normal_rate : float
        正常人参考运动速率, 用于计算速度变慢的程度。

    Returns
    -------
    dict
    """
    if features is None or not isinstance(features, dict):
        return _error("features 是 None 或不是 dict")

    missing = [k for k in REQUIRED_FEATURE_KEYS if k not in features]
    if missing:
        return _error(f"缺少特征键: {missing}")

    median_movement_rate = features["median_movement_rate"]

    # 边界: 速率缺失或 0 -> 强制 4 分, 避免除零
    if _is_missing(median_movement_rate) or median_movement_rate == 0:
        return {
            "final_score":       4,
            "speed_score":       None,
            "speed_level":       "无法评估",
            "speed_reason":      "median_movement_rate 缺失或为 0",
            "pause_score":       0,
            "decay_score":       0,
            "base_score":        0,
            "special_rule":      "无有效事件 -> 强制 4 分",
            "rule_triggered":    "",
            "score_explanation": "median_movement_rate 不可用, 强制最终得分=4",
            "status":            "ok",
        }

    rule_details = []

    # ========= 1) 速度评估 =========
    R = float(normal_rate / median_movement_rate)
    if R < 1.2:
        speed_score = 0; speed_level = "正常";    speed_reason = f"R={R:.2f} <1.2"
    elif R < 1.8:
        speed_score = 1; speed_level = "轻微变慢"; speed_reason = f"1.2<=R={R:.2f}<1.8"
    elif R < 2.4:
        speed_score = 2; speed_level = "轻度变慢"; speed_reason = f"1.8<=R={R:.2f}<2.4"
    else:
        speed_score = 3; speed_level = "中度变慢"; speed_reason = f"R={R:.2f}>=2.4"

    pause_count   = features["num_pauses"]
    freeze_count  = features["num_freezes"]
    has_decay     = bool(features["decay_occurred"])
    decay_start   = features["first_decay_position"]
    event_count   = features["num_events"]
    max_amplitude = features["max_amplitude"]

    symptom_scores = []

    # ========= 2) 暂停 / 冻结 =========
    pause_score = 0
    if pause_count > 5 or freeze_count >= 1:
        pause_score = 3; rule_details.append("Pause>5 or Freeze>=1 -> 3 分")
    elif 3 <= pause_count <= 5:
        pause_score = 2; rule_details.append("Pause 3-5 -> 2 分")
    elif 1 <= pause_count <= 2:
        pause_score = 1; rule_details.append("Pause 1-2 -> 1 分")
    if pause_score > 0:
        symptom_scores.append(pause_score)

    # ========= 3) 衰减 =========
    decay_score = 0
    if has_decay and not _is_missing(decay_start):
        if decay_start <= 2:
            decay_score = 3; rule_details.append("Early decay (<=2) -> 3 分")
        elif 3 <= decay_start <= 6:
            decay_score = 2; rule_details.append("Mid decay (3-6) -> 2 分")
        elif decay_start in (7, 8):
            decay_score = 1; rule_details.append("Late decay (7-8) -> 1 分")
    if decay_score > 0:
        symptom_scores.append(decay_score)

    # ========= 4) 速度并入 =========
    if speed_score > 0:
        symptom_scores.append(speed_score)
        rule_details.append(f"Speed {speed_level} -> {speed_score} 分")

    base_score = max(symptom_scores) if symptom_scores else 0

    # ========= 5) 特殊情况 =========
    special_rule = "无"
    if event_count in (6, 7):
        final_score = base_score + 1
        special_rule = "事件数目异常 -> +1"
    elif event_count in (2, 5):
        final_score = 4 if base_score > 0 else 3
        special_rule = "无法完成实验 -> 强制 3/4 分"
    else:
        final_score = base_score
    final_score = min(final_score, 4)

    explanation = (
        f"速度评估: {speed_level} ({speed_reason}); "
        f"暂停次数={pause_count}, 冻结次数={freeze_count}; "
        f"衰减={has_decay}, 起始位置={decay_start}; "
        f"基础分={base_score}; "
        f"特殊规则={special_rule}; "
        f"最终得分={final_score}"
    )

    return {
        "final_score":       int(final_score),
        "speed_score":       int(speed_score),
        "speed_level":       speed_level,
        "speed_reason":      speed_reason,
        "pause_score":       int(pause_score),
        "decay_score":       int(decay_score),
        "base_score":        int(base_score),
        "special_rule":      special_rule,
        "rule_triggered":    " | ".join(rule_details),
        "score_explanation": explanation,
        "status":            "ok",
    }


def _error(msg):
    return {
        "final_score":       None,
        "speed_score":       None,
        "speed_level":       None,
        "speed_reason":      None,
        "pause_score":       None,
        "decay_score":       None,
        "base_score":        None,
        "special_rule":      None,
        "rule_triggered":    "",
        "score_explanation": "",
        "status":            f"error: {msg}",
    }

# ============================================================
# >>>>>>>>>>>>>>>>>>  DEBUG / DEV-ONLY  >>>>>>>>>>>>>>>>>>>>>>
# ============================================================
if __name__ == "__main__":
    import json
    DEBUG_FEATURES_JSON = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\b_finger-tapping\d_rate\debug_features.json"
    DEBUG_OUTPUT_JSON = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\b_finger-tapping\d_rate\debug_score_based-on-rules.json"
    left_or_right = "left"
    if left_or_right == "left":
        DEBUG_NORMAL_RATE = 2.103
    else:
        DEBUG_NORMAL_RATE = 2.514

    print(f"[DEBUG] 读取特征 JSON: {DEBUG_FEATURES_JSON}")
    with open(DEBUG_FEATURES_JSON, "r", encoding="utf-8") as f:
        features = json.load(f)

    print("[DEBUG] 加载特征:")
    print(json.dumps(features, indent=2, ensure_ascii=False))

    print(f"\n[DEBUG] 打分 (normal_rate={DEBUG_NORMAL_RATE})")
    result = score_finger_tapping(features, normal_rate=DEBUG_NORMAL_RATE)
    print("[DEBUG] 结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if DEBUG_OUTPUT_JSON:
        os.makedirs(os.path.dirname(DEBUG_OUTPUT_JSON) or ".", exist_ok=True)
        with open(DEBUG_OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[DEBUG] 打分 JSON: {DEBUG_OUTPUT_JSON}")
# ============================================================
# <<<<<<<<<<<<<<<<<<  END DEBUG / DEV-ONLY  <<<<<<<<<<<<<<<<<<
# ============================================================