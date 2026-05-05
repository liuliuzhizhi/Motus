"""
============================================================
Arising-from-chair  Step 5 / 5 - Rule-based scoring
============================================================
Backend usage:
    from step5_arising_rule_scoring import score_arising

    result = score_arising(events_df, hand_support_df, reference_speed=0.5)

Inputs
------
events_df : pd.DataFrame
    Output of step 3.
hand_support_df : pd.DataFrame
    Output of step 4 (rows = successful events with `any_hand_support`
    among other columns). Empty DataFrame is allowed.
reference_speed : float or None
    Reference stand_speed used to detect "slow" rising. Pass None to
    disable the speed criterion entirely.

Returns
-------
dict (JSON-serialisable). Always contains `final_score` and `status`.
============================================================
"""
import os
import math
import numpy as np
import pandas as pd


# ============================================================
# Tunable thresholds (preserved from the original)
# ============================================================
STABLE_DURATION_THRESHOLD = 0.5   # seconds; success must stay up at least this long
MULTIPLE_FAILURE_COUNT    = 1     # >= this many failures before success = "multiple"
SPEED_RATIO_THRESHOLD     = 0.8   # speed < ratio * reference  =>  slow


# ============================================================
# Helpers
# ============================================================
def _is_missing(v):
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return False


def _classify_event(row, stable_duration_threshold=STABLE_DURATION_THRESHOLD):
    cat = str(row.get("category", "")).strip()
    dur = row.get("stand_duration", np.nan)
    if cat != "Successful":
        return "failure"
    if not np.isfinite(dur) or dur < stable_duration_threshold:
        return "failure"
    return "success"


def _build_hand_support_lookup(hand_support_df):
    """
    Returns dict[int -> bool] mapping event_id -> any_hand_support.
    Missing keys mean "no record; assume not used".
    """
    if hand_support_df is None or len(hand_support_df) == 0:
        return {}
    if "event_id" not in hand_support_df.columns:
        return {}
    if "any_hand_support" not in hand_support_df.columns:
        return {}
    df = hand_support_df.dropna(subset=["event_id"]).copy()
    df["event_id"] = df["event_id"].astype(int)
    return dict(zip(df["event_id"], df["any_hand_support"].astype(bool)))


# ============================================================
# Main entry
# ============================================================
def score_arising(events_df,
                  hand_support_df=None,
                  reference_speed=0.5,
                  stable_duration_threshold=STABLE_DURATION_THRESHOLD,
                  multiple_failure_count=MULTIPLE_FAILURE_COUNT,
                  speed_ratio_threshold=SPEED_RATIO_THRESHOLD):
    """
    Apply the rule-based MDS-UPDRS 3.9 scoring to a single recording.

    Returns
    -------
    dict
    """
    if events_df is None:
        return _error("events_df is None")
    if not isinstance(events_df, pd.DataFrame):
        return _error(f"events_df should be DataFrame, got {type(events_df).__name__}")

    df = events_df.copy()
    if len(df) > 0:
        df = df.sort_values("event_id").reset_index(drop=True)
        df["classification"] = df.apply(
            lambda r: _classify_event(r, stable_duration_threshold), axis=1
        )
        success_rows = df[df["classification"] == "success"]
        n_fail_total = int((df["classification"] == "failure").sum())
    else:
        success_rows = df.iloc[0:0]
        n_fail_total = 0

    hs_lookup = _build_hand_support_lookup(hand_support_df)

    rec = {
        "n_events_total":    int(len(df)),
        "n_failures_total":  n_fail_total,
        "n_successes_total": int(len(success_rows)),
        "reference_speed":   reference_speed,
        "status":            "ok",
    }

    # ---- No stable success at all -> 4 ----
    if len(success_rows) == 0:
        rec.update({
            "n_failures_before_first_success": n_fail_total,
            "first_success_event_id":          None,
            "first_success_used_hand_support": None,
            "first_success_stand_speed":       None,
            "first_success_stand_duration":    None,
            "speed_ratio":                     None,
            "decision_branch":                 "no_stable_success",
            "final_score":                     4,
            "rationale": (f"No stable success in {len(df)} events "
                          f"({n_fail_total} failed). -> 4"),
        })
        return rec

    first      = success_rows.iloc[0]
    first_eid  = int(first["event_id"])
    n_fail_bef = int(((df["classification"] == "failure") &
                      (df["event_id"] < first_eid)).sum())

    if first_eid in hs_lookup:
        used_support = bool(hs_lookup[first_eid])
        hs_note = ""
    else:
        used_support = False
        hs_note = " (no hand-support record; assumed not used)"

    speed = first.get("stand_speed", np.nan)
    speed = float(speed) if speed is not None and np.isfinite(speed) else float("nan")
    if (reference_speed is not None and reference_speed > 0
            and np.isfinite(speed)):
        speed_ratio = speed / float(reference_speed)
    else:
        speed_ratio = float("nan")

    multiple = n_fail_bef >= multiple_failure_count
    slow     = np.isfinite(speed_ratio) and (speed_ratio < speed_ratio_threshold)

    if used_support:
        if multiple:
            score, branch = 3, "support_with_multiple_failures"
            reason = (f"First stable success at event {first_eid} used hand support; "
                      f"{n_fail_bef} prior failures "
                      f"(>= {multiple_failure_count}). -> 3")
        else:
            score, branch = 2, "support_no_multiple_failures"
            reason = (f"First stable success at event {first_eid} used hand support; "
                      f"{n_fail_bef} prior failure(s) "
                      f"(< {multiple_failure_count}). -> 2")
    else:
        if multiple:
            score, branch = 1, "no_support_with_multiple_failures"
            reason = (f"First stable success at event {first_eid}, no hand support; "
                      f"{n_fail_bef} prior failures "
                      f"(>= {multiple_failure_count}). -> 1")
        elif slow:
            score, branch = 1, "no_support_slow_speed"
            reason = (f"First stable success at event {first_eid}, no hand support; "
                      f"speed {speed:.3f} < {speed_ratio_threshold} x ref"
                      f"({reference_speed:.3f}); ratio={speed_ratio:.2f}. -> 1")
        else:
            score, branch = 0, "normal"
            spd_str = (f", speed {speed:.3f} (ratio={speed_ratio:.2f})"
                       if np.isfinite(speed_ratio) else "")
            reason = (f"First stable success at event {first_eid}, no hand support, "
                      f"{n_fail_bef} prior failure(s){spd_str}. -> 0")

    rec.update({
        "n_failures_before_first_success": n_fail_bef,
        "first_success_event_id":          first_eid,
        "first_success_used_hand_support": used_support,
        "first_success_stand_speed":       (speed if np.isfinite(speed) else None),
        "first_success_stand_duration":    float(first.get("stand_duration", np.nan))
                                             if np.isfinite(first.get("stand_duration", np.nan)) else None,
        "speed_ratio":                     float(speed_ratio) if np.isfinite(speed_ratio) else None,
        "decision_branch":                 branch,
        "final_score":                     int(score),
        "rationale":                       reason + hs_note,
    })
    return rec


def _error(msg):
    return {
        "n_events_total":    None,
        "n_failures_total":  None,
        "n_successes_total": None,
        "reference_speed":   None,
        "final_score":       None,
        "decision_branch":   None,
        "rationale":         "",
        "status":            f"error: {msg}",
    }


# ============================================================
# >>>>>>>>>>>>>>>>>>  DEBUG / DEV-ONLY  >>>>>>>>>>>>>>>>>>>>>>
# Edit the paths below and run:  python step5_arising_rule_scoring.py
# Delete this whole section for production - nothing above depends on it.
# ============================================================
if __name__ == "__main__":
    import json
    DEBUG_EVENTS_CSV = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\h_arising-from-chair\c_action-recognition\arising_event_detection.csv"
    DEBUG_HAND_SUPPORT_CSV = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\h_arising-from-chair\c_action-recognition\debug_hand_support.csv"
    DEBUG_OUTPUT_JSON = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\h_arising-from-chair\d_rate\score_based-on-rules.json"
    DEBUG_REFERENCE_SPEED = 0.5

    print(f"[DEBUG] Reading events       : {DEBUG_EVENTS_CSV}")
    events = pd.read_csv(DEBUG_EVENTS_CSV)
    print(f"[DEBUG] Reading hand-support : {DEBUG_HAND_SUPPORT_CSV}")
    try:
        hs = pd.read_csv(DEBUG_HAND_SUPPORT_CSV)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        hs = pd.DataFrame()
        print(f"[DEBUG] Hand-support file empty/missing - using empty table")

    print(f"[DEBUG] Scoring (ref_speed={DEBUG_REFERENCE_SPEED})")
    result = score_arising(events, hs, reference_speed=DEBUG_REFERENCE_SPEED)
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