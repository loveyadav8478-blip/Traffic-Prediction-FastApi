import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

MODEL_PATH   = "model_XG.pkl"
COLUMNS_PATH = "columns.pkl"

TRAFFIC_LABELS = {
    0: "Low",
    1: "Medium",
    2: "High"
}

AVG_SPEED_KMH = {
    0: 80.0,
    1: 50.0,
    2: 25.0
}

# All possible holiday values from your training data
ALL_HOLIDAYS = [
    "Columbus Day",
    "Independence Day",
    "Labor Day",
    "Martin Luther King Jr Day",
    "Memorial Day",
    "New Years Day",
    "No Holiday",
    "State Fair",
    "Thanksgiving Day",
    "Veterans Day",
    "Washingtons Birthday"
]

# All possible weather values from your training data
ALL_WEATHER = [
    "Clouds",
    "Drizzle",
    "Fog",
    "Haze",
    "Mist",
    "Rain",
    "Smoke",
    "Snow",
    "Squall",
    "Thunderstorm"
]

# ─────────────────────────────────────────────────────────────
# LOAD MODEL + COLUMNS
# ─────────────────────────────────────────────────────────────

def _load():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ '{MODEL_PATH}' not found.")
    if not os.path.exists(COLUMNS_PATH):
        raise FileNotFoundError(f"❌ '{COLUMNS_PATH}' not found.")

    model   = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    print(f"✅ XGBoost model loaded. Expects {len(columns)} features.")
    return model, columns

_model, _columns = _load()


# ─────────────────────────────────────────────────────────────
# KNOWN DISTANCES (Indian cities) — km
# Nominatim geocoding will replace this in next feature
# ─────────────────────────────────────────────────────────────

KNOWN_DISTANCES = {
    frozenset(["delhi", "agra"])        : 233,
    frozenset(["delhi", "jaipur"])      : 281,
    frozenset(["delhi", "mumbai"])      : 1415,
    frozenset(["mumbai", "pune"])       : 149,
    frozenset(["bangalore", "mysore"])  : 145,
    frozenset(["chennai", "bangalore"]) : 346,
    frozenset(["delhi", "chandigarh"])  : 274,
    frozenset(["agra", "firozabad"])    : 40,
    frozenset(["delhi", "firozabad"])   : 198,
    frozenset(["agra", "lucknow"])      : 331,
    frozenset(["mumbai", "nashik"])     : 167,
    frozenset(["delhi", "lucknow"])     : 555,
    frozenset(["hyderabad", "bangalore"]): 569,
}

def estimate_distance(source: str, destination: str) -> float:
    key = frozenset([source.lower().strip(), destination.lower().strip()])
    if key in KNOWN_DISTANCES:
        return float(KNOWN_DISTANCES[key])
    # Deterministic fallback — same input always gives same output
    src_val  = sum(ord(c) for c in source.lower())
    dst_val  = sum(ord(c) for c in destination.lower())
    distance = abs(src_val - dst_val) * 1.8 + len(source) * 4.5
    return round(min(max(distance, 20), 800), 1)


# ─────────────────────────────────────────────────────────────
# BUILD FEATURE ROW — matches columns.pkl exactly
# ─────────────────────────────────────────────────────────────

def _build_features(now: datetime, weather: str = "Clouds",
                    holiday: str = "No Holiday",
                    temp_cel: float = 25.0,
                    rain_1h: float = 0.0,
                    snow_1h: float = 0.0,
                    clouds_all: float = 40.0) -> pd.DataFrame:
    """
    Build a single-row DataFrame that matches training columns exactly.
    Default values represent a normal clear weekday afternoon.
    """

    row = {
        # ── Numeric features ──────────────────────────────
        "rain_1h"    : rain_1h,
        "snow_1h"    : snow_1h,
        "clouds_all" : clouds_all,
        "year"       : now.year,
        "month"      : now.month,
        "day"        : now.day,
        "hour"       : now.hour,
        "minute"     : now.minute,
        "day_of_week": now.weekday(),   # 0=Mon, 6=Sun
        "temp_cel"   : temp_cel,

        # ── Holiday one-hot ───────────────────────────────
        **{f"holiday_{h}": (1 if h == holiday else 0)
           for h in ALL_HOLIDAYS},

        # ── Weather one-hot ───────────────────────────────
        **{f"weather_main_{w}": (1 if w == weather else 0)
           for w in ALL_WEATHER},
    }

    df = pd.DataFrame([row])

    # ── Reorder to EXACTLY match training column order ────
    df = df.reindex(columns=_columns, fill_value=0)

    return df


# ─────────────────────────────────────────────────────────────
# MAIN PREDICT FUNCTION
# ─────────────────────────────────────────────────────────────

def predict(source: str, destination: str,
            weather: str    = "Clouds",
            holiday: str    = "No Holiday",
            temp_cel: float = 25.0) -> dict:

    now      = datetime.now()
    distance = estimate_distance(source, destination)

    # Build feature row
    features = _build_features(
        now        = now,
        weather    = weather,
        holiday    = holiday,
        temp_cel   = temp_cel
    )

    # Predict
    traffic_level = int(_model.predict(features)[0])
    probabilities = _model.predict_proba(features)[0]
    confidence    = float(probabilities[traffic_level])

    # Calculate durations
    speed          = AVG_SPEED_KMH[traffic_level]
    free_flow_min  = round((distance / 80.0) * 60, 1)
    real_min       = round((distance / speed) * 60, 1)
    delay_min      = round(max(real_min - free_flow_min, 0), 1)

    return {
        "distance_km"       : distance,
        "duration_min"      : free_flow_min,
        "real_duration_min" : real_min,
        "delay_min"         : delay_min,
        "final_prediction"  : traffic_level,
        "traffic_label"     : TRAFFIC_LABELS[traffic_level],
        "confidence"        : round(confidence, 2),
        "predicted_at"      : now.isoformat(),
        "features_used"     : {
            "hour"       : now.hour,
            "day_of_week": now.weekday(),
            "weather"    : weather,
            "holiday"    : holiday,
            "temp_cel"   : temp_cel,
            "distance_km": distance
        }
    }