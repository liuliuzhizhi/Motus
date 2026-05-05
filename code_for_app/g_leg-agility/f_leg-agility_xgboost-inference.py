"""
============================================================
MDS-UPDRS 3.8 Leg Agility - Inference module (app backend)
============================================================
Production usage:
    from predict import LegAgilityScorer

    # Load once at app startup
    scorer = LegAgilityScorer("path/to/leg_agility_model.joblib")

    # Per request
    result = scorer.score(df_events)              # if you have a DataFrame
    result = scorer.score_from_csv(csv_path)      # if you have a file path

`result` is a JSON-serialisable dict, e.g.:
    {
        "score":         2,
        "confidence":    0.78,
        "is_rule":       False,
        "probabilities": {0: 0.05, 1: 0.10, 2: 0.78, 3: 0.05, 4: 0.02},
        "n_events_used": 8,
        "status":        "ok"
    }

`status` values:
    "ok"            - model prediction succeeded
    "ok_rule"       - rule-based fallback (no valid events)
    "error: ..."    - input problem; `score` is None, do NOT use it
============================================================
"""
import os
import joblib
import numpy as np
import pandas as pd

# Required columns in the events CSV/DataFrame
REQUIRED_EVENT_COLS = (
    "event_index", "start_frame", "end_frame",
    "rise90_frame", "fall90_frame", "peak_amplitude",
)

# ------------------------------------------------------------
# Same signal builder used during training - DO NOT modify
# without retraining; the model expects this exact shape.
# ------------------------------------------------------------
def _events_to_multichannel_signal(df_events, max_events):
    """Event table -> 3-channel square-wave signal (amplitude / rising / falling)."""
    if len(df_events) == 0:
        return None
    df = df_events.sort_values("event_index").reset_index(drop=True)
    if len(df) > max_events:
        df = df.iloc[:max_events].copy()
    t0 = int(df["start_frame"].min())
    T  = int(df["end_frame"].max()) - t0 + 1
    if T <= 0:
        return None
    amp_ch  = np.zeros(T, dtype=np.float32)
    rise_ch = np.zeros(T, dtype=np.float32)
    fall_ch = np.zeros(T, dtype=np.float32)
    for _, row in df.iterrows():
        s  = max(0, int(row["start_frame"])  - t0)
        e  = min(T - 1, int(row["end_frame"]) - t0)
        r9 = max(s, min(e, int(row["rise90_frame"]) - t0))
        f9 = max(r9, min(e, int(row["fall90_frame"]) - t0))
        amp = float(row["peak_amplitude"])
        amp_ch [s  : e  + 1] = amp
        rise_ch[s  : r9 + 1] = amp
        fall_ch[f9 : e  + 1] = amp
    return np.stack([amp_ch, rise_ch, fall_ch], axis=0)

# ------------------------------------------------------------
# Scorer class - load once, score many times
# ------------------------------------------------------------
class LegAgilityScorer:
    """
    Wrapper around the saved model bundle. Instantiate once at app
    startup and reuse across requests. Methods are read-only and
    safe to share across worker threads.
    """

    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        bundle = joblib.load(model_path)
        self._mr  = bundle["minirocket"]
        self._sc  = bundle["scaler"]
        self._sel = bundle["selector"]
        self._clf = bundle["classifier"]
        self.config  = bundle["config"]
        self.classes = list(self._clf.classes_)

    # --------------------------------------------------------
    # Main API: score from an in-memory DataFrame
    # --------------------------------------------------------
    def score(self, df_events):
        """
        Parameters
        ----------
        df_events : pd.DataFrame
            Event table with columns listed in REQUIRED_EVENT_COLS.
            Empty DataFrame is allowed - triggers rule-based fallback.

        Returns
        -------
        dict (JSON-serialisable). Always contains the key `status`.
        Callers should branch on `status` before trusting `score`.
        """
        # ---- Validate input ----
        if df_events is None:
            return self._error_result("input is None")
        if not isinstance(df_events, pd.DataFrame):
            return self._error_result(
                f"expected pandas DataFrame, got {type(df_events).__name__}"
            )
        if len(df_events) > 0:
            missing = [c for c in REQUIRED_EVENT_COLS if c not in df_events.columns]
            if missing:
                return self._error_result(f"missing columns {missing}")

        # ---- Build signal ----
        try:
            sig = _events_to_multichannel_signal(
                df_events, max_events=self.config["MAX_EVENTS"]
            )
        except (KeyError, ValueError, TypeError) as e:
            return self._error_result(f"signal construction failed: {e}")

        # ---- Rule-based fallback: no usable event ----
        if sig is None:
            return {
                "score":         int(self.config["RULE_PRED"]),
                "confidence":    1.0,
                "is_rule":       True,
                "probabilities": None,
                "n_events_used": 0,
                "status":        "ok_rule",
            }

        # ---- Run trained pipeline ----
        try:
            n_events_used = min(len(df_events), self.config["MAX_EVENTS"])
            X_nested = pd.DataFrame({
                f"dim_{c}": [pd.Series(sig[c])] for c in range(sig.shape[0])
            })
            Xt = np.asarray(self._mr.transform(X_nested), dtype=np.float32)
            Xt = self._sc.transform(Xt)
            Xs = self._sel.transform(Xt)

            proba = self._clf.predict_proba(Xs)[0]
            pred  = int(self._clf.classes_[int(np.argmax(proba))])
        except Exception as e:
            return self._error_result(f"inference failed: {e}")

        return {
            "score":         pred,
            "confidence":    float(proba.max()),
            "is_rule":       False,
            "probabilities": {int(c): float(p)
                              for c, p in zip(self._clf.classes_, proba)},
            "n_events_used": int(n_events_used),
            "status":        "ok",
        }

    # --------------------------------------------------------
    # Convenience: load CSV then score
    # --------------------------------------------------------
    def score_from_csv(self, csv_path):
        if not os.path.exists(csv_path):
            return self._error_result(f"csv not found: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            df = pd.DataFrame()
        except Exception as e:
            return self._error_result(f"csv read failed: {e}")
        return self.score(df)

    # --------------------------------------------------------
    # Internal: uniform error payload
    # --------------------------------------------------------
    @staticmethod
    def _error_result(msg):
        return {
            "score":         None,
            "confidence":    None,
            "is_rule":       False,
            "probabilities": None,
            "n_events_used": 0,
            "status":        f"error: {msg}",
        }


# ============================================================
# >>>>>>>>>>>>>>>>>>  DEBUG / DEV-ONLY  >>>>>>>>>>>>>>>>>>>>>>
# Everything below this banner is a local-debug harness.
# The app backend imports `LegAgilityScorer` directly and never
# touches this section.
#
# TO REMOVE FOR PRODUCTION:
#   delete from this banner down to the matching "<<<<<<" banner.
#   Nothing above it will be affected.
# ============================================================
if __name__ == "__main__":
    import json

    # ---- Edit these paths to test locally ----
    Left_or_right = "left"
    if Left_or_right == "left":
        DEBUG_MODEL_PATH = "g_leg-agility\leg-agility-L_model.joblib"
    else:
        DEBUG_MODEL_PATH = "g_leg-agility\leg-agility-R_model.joblib"
    DEBUG_INPUT_CSV   = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\leg-agility\c_event-detection\debug_events.csv"
    DEBUG_OUTPUT_JSON = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\leg-agility\d_rate\score_model-predict.json"   # set to None to skip writing

    print("=" * 60)
    print("[DEBUG] Loading model...")
    scorer = LegAgilityScorer(DEBUG_MODEL_PATH)
    print(f"[DEBUG] Classes: {scorer.classes}")
    print(f"[DEBUG] Config : {scorer.config}")

    print(f"\n[DEBUG] Scoring file: {DEBUG_INPUT_CSV}")
    result = scorer.score_from_csv(DEBUG_INPUT_CSV)

    print("\n[DEBUG] Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Pretty bar chart of class probabilities
    if result.get("probabilities"):
        print("\n[DEBUG] Per-class probabilities:")
        for c, p in sorted(result["probabilities"].items()):
            bar = "#" * int(round(p * 30))
            print(f"  score={c} | {p:.4f} | {bar}")

    # Optional: persist result for inspection
    if DEBUG_OUTPUT_JSON:
        os.makedirs(os.path.dirname(DEBUG_OUTPUT_JSON) or ".", exist_ok=True)
        with open(DEBUG_OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[DEBUG] Wrote result to {DEBUG_OUTPUT_JSON}")
# ============================================================
# <<<<<<<<<<<<<<<<<<  END DEBUG / DEV-ONLY  <<<<<<<<<<<<<<<<<<
# ============================================================