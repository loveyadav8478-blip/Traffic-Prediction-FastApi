"""
predictor.py — Routex Indian Traffic Prediction Engine v2
Replaces original predictor.py — auth.py and .env unchanged
"""
import os
import logging
import requests
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load model ─────────────────────────────────────────────
print("🔄 Loading Routex Indian traffic model...")
model        = joblib.load(os.path.join(BASE_DIR, 'india_traffic_model.pkl'))
feature_cols = joblib.load(os.path.join(BASE_DIR, 'feature_columns.pkl'))
print(f"✅ Model loaded — {len(feature_cols)} features")

# ── City mapping (40+ Indian cities → model cities) ────────
CITY_MAP = {
    # Delhi NCR
    "delhi": "Delhi", "new delhi": "Delhi", "ghaziabad": "Delhi",
    "noida": "Delhi", "faridabad": "Delhi", "gurugram": "Delhi",
    "gurgaon": "Delhi", "meerut": "Delhi", "sonipat": "Delhi",
    "rohtak": "Delhi", "panipat": "Delhi", "hapur": "Delhi",
    # UP
    "agra": "Agra", "mathura": "Agra", "firozabad": "Agra",
    "tundla": "Agra", "etawah": "Agra", "mainpuri": "Agra",
    "kanpur": "Lucknow", "lucknow": "Lucknow", "unnao": "Lucknow",
    "varanasi": "Lucknow", "prayagraj": "Lucknow", "allahabad": "Lucknow",
    "gorakhpur": "Lucknow", "bareilly": "Lucknow", "aligarh": "Agra",
    # Rajasthan
    "jaipur": "Jaipur", "jodhpur": "Jaipur", "udaipur": "Jaipur",
    "kota": "Jaipur", "ajmer": "Jaipur", "bikaner": "Jaipur",
    "alwar": "Jaipur", "bharatpur": "Jaipur",
    # Maharashtra
    "mumbai": "Mumbai", "thane": "Mumbai", "navi mumbai": "Mumbai",
    "kalyan": "Mumbai", "vasai": "Mumbai",
    "pune": "Pune", "nashik": "Pune", "aurangabad": "Pune",
    "solapur": "Pune", "kolhapur": "Pune",
    # South
    "bangalore": "Bangalore", "bengaluru": "Bangalore",
    "mysore": "Bangalore", "mysuru": "Bangalore",
    "mangalore": "Bangalore", "hubli": "Bangalore",
    "hyderabad": "Hyderabad", "secunderabad": "Hyderabad",
    "warangal": "Hyderabad", "vijayawada": "Hyderabad",
    "visakhapatnam": "Hyderabad", "vizag": "Hyderabad",
    "chennai": "Chennai", "coimbatore": "Chennai",
    "madurai": "Chennai", "trichy": "Chennai",
    "salem": "Chennai", "tirunelveli": "Chennai",
    # East
    "kolkata": "Kolkata", "howrah": "Kolkata", "durgapur": "Kolkata",
    "patna": "Kolkata", "bhubaneswar": "Kolkata", "cuttack": "Kolkata",
    "ranchi": "Kolkata", "jamshedpur": "Kolkata",
    # Others
    "chandigarh": "Delhi", "ludhiana": "Delhi", "amritsar": "Delhi",
    "surat": "Mumbai", "vadodara": "Mumbai", "ahmedabad": "Mumbai",
    "bhopal": "Lucknow", "indore": "Pune", "nagpur": "Pune",
    "guwahati": "Kolkata", "siliguri": "Kolkata",
}

# ── Weather mapping ────────────────────────────────────────
WEATHER_MAP = {
    "clear": "Clear", "sunny": "Clear", "hot": "Clear",
    "cloudy": "Clouds", "clouds": "Clouds", "overcast": "Clouds",
    "partly cloudy": "Clouds", "windy": "Clouds",
    "rainy": "Rain", "rain": "Rain", "raining": "Rain",
    "drizzle": "Drizzle", "light rain": "Drizzle",
    "fog": "Fog", "foggy": "Fog",
    "haze": "Haze", "hazy": "Haze",
    "mist": "Mist", "misty": "Mist",
    "thunderstorm": "Thunderstorm", "thunder": "Thunderstorm",
    "storm": "Thunderstorm", "lightning": "Thunderstorm",
    "smoke": "Smoke", "smog": "Smoke", "pollution": "Smoke",
    "snowy": "Clear", "snow": "Clear",
    "sandstorm": "Haze", "dust": "Haze",
}

# ── Indian holidays ────────────────────────────────────────
INDIAN_HOLIDAYS = {
    (1, 1):  "New Years Day",
    (1, 26): "Republic Day",
    (8, 15): "Independence Day",
    (10, 2): "Dussehra",
    (12, 25): "Christmas",
}

# Festival season months — higher base probability
FESTIVAL_MONTHS = {10, 11}  # Oct-Nov: Navratri, Dussehra, Diwali, Eid


def map_city(name: str) -> str:
    return CITY_MAP.get((name or "").lower().strip(), "Delhi")


def map_weather(w: str) -> str:
    return WEATHER_MAP.get((w or "").lower().strip(), "Clear")


def label_from_int(val: int) -> str:
    return {0: "Low", 1: "Medium", 2: "High"}.get(int(val), "Low")


def get_osrm_distance(source: str, destination: str) -> Optional[float]:
    """Fetch real road distance via Nominatim + OSRM."""
    try:
        def geocode(city: str):
            r = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": f"{city}, India", "format": "json", "limit": 1},
                headers={"User-Agent": "Routex/2.0"},
                timeout=6,
            )
            data = r.json()
            if data:
                return float(data[0]["lon"]), float(data[0]["lat"])
            return None

        src = geocode(source)
        dst = geocode(destination)
        if not src or not dst:
            return None

        r = requests.get(
            f"https://router.project-osrm.org/route/v1/driving/"
            f"{src[0]},{src[1]};{dst[0]},{dst[1]}",
            params={"overview": "false"},
            timeout=8,
        )
        data = r.json()
        if data.get("code") == "Ok":
            return round(data["routes"][0]["distance"] / 1000, 1)
    except Exception as e:
        logger.warning("OSRM distance fetch failed: %s", e)
    return None


def get_season_temp(month: int, city: str) -> float:
    """Estimate temperature by month and city region."""
    north = {"Delhi", "Agra", "Lucknow", "Jaipur"}
    south = {"Chennai", "Hyderabad", "Bangalore"}

    if month in [4, 5, 6]:
        return 42.0 if city in north else 35.0
    elif month in [12, 1, 2]:
        return 12.0 if city in north else 24.0
    elif month in [7, 8, 9]:
        return 30.0 if city in north else 28.0
    return 28.0


def build_features(
    hour: int, dow: int, weather: str, city: str,
    holiday: str, distance_km: float, temp: float, month: int
) -> pd.DataFrame:
    """Build feature vector matching trained model columns exactly."""
    row = {col: 0 for col in feature_cols}

    row["hour"]                  = hour
    row["day_of_week"]           = dow
    row["is_weekend"]            = 1 if dow >= 5 else 0
    row["is_peak_hour"]          = 1 if (8 <= hour <= 10 or 18 <= hour <= 20) else 0
    row["temp_cel"]              = temp
    row["distance_km"]           = distance_km
    row["month"]                 = month
    row["year"]                  = datetime.now().year
    row["day"]                   = datetime.now().day
    row["minute"]                = 0
    row["rain_1h"]               = 5.0 if weather in ["Rain","Thunderstorm","Drizzle"] else 0.0
    row["snow_1h"]               = 0.0
    row["clouds_all"]            = 85 if weather in ["Clouds","Rain","Thunderstorm"] else 15
    row["signal_count"]          = 14 if city in ["Delhi","Mumbai","Bangalore"] else 8
    row["traffic_noise_factor"]  = 1.2 if city in ["Delhi","Mumbai","Bangalore","Kolkata"] else 1.0

    # One-hot: weather
    wkey = f"weather_{weather}"
    if wkey in row:
        row[wkey] = 1
    elif "weather_Clear" in row:
        row["weather_Clear"] = 1

    # One-hot: city
    ckey = f"city_{city}"
    if ckey in row:
        row[ckey] = 1
    elif "city_Delhi" in row:
        row["city_Delhi"] = 1

    # One-hot: holiday
    hkey = f"holiday_{holiday}"
    if hkey in row:
        row[hkey] = 1
    elif "holiday_No Holiday" in row:
        row["holiday_No Holiday"] = 1

    # One-hot: vehicle density (medium default)
    if "vehicle_density_medium" in row:
        row["vehicle_density_medium"] = 1

    # One-hot: road type (city default)
    if "road_type_city" in row:
        row["road_type_city"] = 1

    return pd.DataFrame([row])[feature_cols]


def predict_traffic(
    source:      str,
    destination: str,
    hour:        Optional[int]   = None,
    day_of_week: Optional[int]   = None,
    weather:     Optional[str]   = "Clear",
    holiday:     Optional[str]   = None,
    distance_km: Optional[float] = None,
    temp_cel:    Optional[float] = None,
) -> dict:
    """
    Main prediction function called from main.py
    Returns best_route dict matching original response shape.
    """
    now   = datetime.now()
    hour  = hour        if hour        is not None else now.hour
    dow   = day_of_week if day_of_week is not None else now.weekday()
    month = now.month

    # Map inputs
    city    = map_city(source)
    weather = map_weather(weather or "Clear")
    temp    = temp_cel if temp_cel else get_season_temp(month, city)

    # Auto-fetch distance if not provided
    if not distance_km or distance_km <= 0:
        logger.info("Fetching road distance: %s → %s", source, destination)
        distance_km = get_osrm_distance(source, destination) or 30.0
    distance_km = round(distance_km, 1)

    # Detect Indian holiday
    if not holiday or holiday == "No Holiday":
        holiday = INDIAN_HOLIDAYS.get((month, now.day), "No Holiday")
        # Festival season boost
        if month in FESTIVAL_MONTHS and holiday == "No Holiday":
            import random
            if random.random() < 0.3:
                holiday = "Diwali" if month == 11 else "Dussehra"

    # Build features and predict
    features      = build_features(hour, dow, weather, city, holiday, distance_km, temp, month)
    pred_int      = int(model.predict(features)[0])
    proba         = model.predict_proba(features)[0]
    confidence    = round(float(max(proba)), 2)
    traffic_label = label_from_int(pred_int)

    # Duration estimate (base speed 50 km/h, adjusted by traffic)
    base_duration = round((distance_km / 50) * 60, 1)
    multiplier    = {"Low": 1.0, "Medium": 1.55, "High": 2.3}[traffic_label]
    real_duration = round(base_duration * multiplier, 1)
    delay         = round(real_duration - base_duration, 1)

    logger.info(
        "Result | %s→%s | city=%s | weather=%s | hour=%d | dow=%d | dist=%.1f km | %s (%.0f%%)",
        source, destination, city, weather, hour, dow, distance_km, traffic_label, confidence * 100
    )

    return {
        "distance_km":       distance_km,
        "duration_min":      base_duration,
        "real_duration_min": real_duration,
        "delay_min":         delay,
        "final_prediction":  pred_int,
        "traffic_label":     traffic_label,
        "confidence":        confidence,
        "predicted_at":      now.isoformat(),
        "features_used": {
            "hour":        hour,
            "day_of_week": dow,
            "is_peak_hour": 1 if (8 <= hour <= 10 or 18 <= hour <= 20) else 0,
            "is_weekend":  1 if dow >= 5 else 0,
            "weather":     weather,
            "holiday":     holiday,
            "temp_cel":    temp,
            "distance_km": distance_km,
            "city":        city,
        },
    }