"""
============================================================
Step 5 / 5 - Rule-based scoring (single patient)
============================================================
Backend usage:
    from step5_rule_scoring import score_from_features

    result = score_from_features(features, normal_rate=267)

`features` is the dict returned by step4_feature_extraction.extract_features.
`result` is a JSON-serialisable dict:
    {
        "final_score":       int,
        "speed_score":       int,
        "speed_level":       str,
        "speed_reason":      str,
        "pause_score":       int,
        "decay_score":       int,
        "base_score":        int,
        "special_rule":      str,
        "rule_triggered":    str,
        "score_explanation": str,
        "status":            "ok" | "error: ..."
    }
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


def score_from_features(features, normal_rate=267):
    """
    Apply the rule-based MDS-UPDRS 3.8 scoring on a single patient's
    kinematic features.

    Parameters
    ----------
    features : dict
        Output of `extract_features`. Required keys:
            median_movement_rate, num_pauses, num_freezes,
            decay_occurred, first_decay_position,
            num_events, max_amplitude
    normal_rate : float
        Reference movement rate; the original project uses 267.

    Returns
    -------
    dict
    """
    if features is None or not isinstance(features, dict):
        return _error("features is None or not a dict")

    missing = [k for k in REQUIRED_FEATURE_KEYS if k not in features]
    if missing:
        return _error(f"missing feature keys: {missing}")

    median_movement_rate = features["median_movement_rate"]

    # If the patient has no usable events at all we cannot run the rules
    # meaningfully - return the conservative "cannot complete" score.
    if _is_missing(median_movement_rate) or median_movement_rate == 0:
        return {
            "final_score":       4,
            "speed_score":       None,
            "speed_level":       "无法评估",
            "speed_reason":      "median_movement_rate 缺失或为 0",
            "pause_score":       0,
            "decay_score":       0,
            "base_score":        0,
            "special_rule":      "无有效事件 → 强制 4 分",
            "rule_triggered":    "",
            "score_explanation": "median_movement_rate 不可用，强制最终得分=4",
            "status":            "ok",
        }

    rule_details = []

    # 1) Speed
    R = float(normal_rate / median_movement_rate)
    if R < 1.5:
        speed_score = 0; speed_level = "正常";    speed_reason = f"R={R:.2f} <1.5"
    elif R < 2:
        speed_score = 1; speed_level = "轻微变慢"; speed_reason = f"1.5≤R={R:.2f}<2"
    elif R < 3:
        speed_score = 2; speed_level = "轻度变慢"; speed_reason = f"2≤R={R:.2f}<3"
    else:
        speed_score = 3; speed_level = "中度变慢"; speed_reason = f"R={R:.2f}≥3"

    pause_count   = features["num_pauses"]
    freeze_count  = features["num_freezes"]
    has_decay     = bool(features["decay_occurred"])
    decay_start   = features["first_decay_position"]
    event_count   = features["num_events"]
    max_amplitude = features["max_amplitude"]

    symptom_scores = []

    # ---- pauses / freezes ----
    pause_score = 0
    if pause_count > 5 or freeze_count >= 1:
        pause_score = 3; rule_details.append("Pause>5 or Freeze≥1 → 3分")
    elif 3 <= pause_count <= 5:
        pause_score = 2; rule_details.append("Pause 3-5 → 2分")
    elif 1 <= pause_count <= 2:
        pause_score = 1; rule_details.append("Pause 1-2 → 1分")
    if pause_score > 0:
        symptom_scores.append(pause_score)

    # ---- amplitude decay ----
    decay_score = 0
    if has_decay and not _is_missing(decay_start):
        if decay_start <= 2:
            decay_score = 3; rule_details.append("Early decay (≤2) → 3分")
        elif 3 <= decay_start <= 6:
            decay_score = 2; rule_details.append("Mid decay (3-6) → 2分")
        elif decay_start in (7, 8):
            decay_score = 1; rule_details.append("Late decay (7-8) → 1分")
    if decay_score > 0:
        symptom_scores.append(decay_score)

    # ---- speed ----
    if speed_score > 0:
        symptom_scores.append(speed_score)
        rule_details.append(f"Speed {speed_level} → {speed_score}分")

    base_score = max(symptom_scores) if symptom_scores else 0

    # 3) Special-case overrides
    special_rule = "无"
    if event_count in (6, 7) or (
        not _is_missing(max_amplitude) and max_amplitude < 10
    ):
        final_score = base_score + 1
        special_rule = "事件异常或幅度过小 → +1"
    elif event_count in (2, 5):
        final_score = 4 if base_score > 0 else 3
        special_rule = "无法完成实验 → 强制 3/4 分"
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
# Edit the paths below and run:  python step5_rule_scoring.py
# Delete this whole section for production - nothing above depends on it.
# ============================================================
if __name__ == "__main__":
    import json

    # The features JSON written by step4_feature_extraction.py's debug section
    DEBUG_FEATURES_JSON = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\leg-agility\d_rate\kinematic_features\debug_features.json"
    DEBUG_OUTPUT_JSON = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\leg-agility\d_rate\score_based_on_rules.json"
    Left_or_right = "left"
    if Left_or_right == "left":
        DEBUG_NORMAL_RATE = 267
    else:
        DEBUG_NORMAL_RATE = 334

    print(f"[DEBUG] Reading features JSON: {DEBUG_FEATURES_JSON}")
    with open(DEBUG_FEATURES_JSON, "r", encoding="utf-8") as f:
        features = json.load(f)

    # JSON has no NaN, step4 wrote None for missing values - rule scorer
    # treats None as missing already, no conversion needed.
    print("[DEBUG] Loaded features:")
    print(json.dumps(features, indent=2, ensure_ascii=False))

    print(f"\n[DEBUG] Scoring (normal_rate={DEBUG_NORMAL_RATE})")
    result = score_from_features(features, normal_rate=DEBUG_NORMAL_RATE)
    print("[DEBUG] Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if DEBUG_OUTPUT_JSON:
        os.makedirs(os.path.dirname(DEBUG_OUTPUT_JSON) or ".", exist_ok=True)
        with open(DEBUG_OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[DEBUG] Wrote score JSON: {DEBUG_OUTPUT_JSON}")
# ============================================================
# <<<<<<<<<<<<<<<<<<  END DEBUG / DEV-ONLY  <<<<<<<<<<<<<<<<<<
# ============================================================