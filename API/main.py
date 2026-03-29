from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import datetime
import googlemaps
import os

# -----------------------------
# CONFIG (Environment Variable)
# -----------------------------
API_KEY = "??"

if not API_KEY:
    raise ValueError("❌ API Key missing! Set GOOGLE_MAPS_API_KEY in environment variables.")

app = FastAPI(title="Traffic Route Prediction API 🚗")

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    model = joblib.load("model_XG.pkl")
    columns = joblib.load("columns.pkl")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# -----------------------------
# GOOGLE MAPS CLIENT
# -----------------------------
gmaps = googlemaps.Client(key=API_KEY)

# -----------------------------
# INPUT SCHEMA
# -----------------------------
class RouteInput(BaseModel):
    source: str
    destination: str


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def get_features():
    now = datetime.datetime.now()

    hour = now.hour
    day_of_week = now.weekday()
    month = now.month

    is_peak_hour = 1 if (7 <= hour <= 10 or 16 <= hour <= 19) else 0
    is_weekend = 1 if day_of_week >= 5 else 0

    if month in [12, 1, 2]:
        season = 0
    elif month in [3, 4, 5]:
        season = 1
    elif month in [6, 7, 8]:
        season = 2
    else:
        season = 3

    return hour, day_of_week, is_peak_hour, is_weekend, season


def real_traffic_level(factor):
    if factor < 1.2:
        return 0
    elif factor < 1.5:
        return 1
    else:
        return 2


# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def home():
    return {"message": "Traffic Route Prediction API 🚀"}


@app.post("/predict-route")
def predict_route(data: RouteInput):
    try:
        # -----------------------------
        # GOOGLE MAPS API CALL
        # -----------------------------
        routes = gmaps.directions(
            data.source,
            data.destination,
            mode="driving",
            alternatives=True,
            departure_time=datetime.datetime.now()
        )

        if not routes:
            raise HTTPException(status_code=404, detail="No routes found")

        results = []

        for i, route in enumerate(routes):
            leg = route['legs'][0]

            duration = leg['duration']['value']
            duration_in_traffic = leg.get('duration_in_traffic', {}).get('value', duration)
            distance = leg['distance']['value']

            # -----------------------------
            # REAL TRAFFIC
            # -----------------------------
            traffic_factor = duration_in_traffic / duration
            real_pred = real_traffic_level(traffic_factor)

            # -----------------------------
            # ML FEATURES
            # -----------------------------
            hour, day_of_week, is_peak_hour, is_weekend, season = get_features()

            df = pd.DataFrame([{
                "temp": 25,
                "rain_1h": 0,
                "snow_1h": 0,
                "clouds_all": 40,
                "year": datetime.datetime.now().year,
                "month": datetime.datetime.now().month,
                "day": datetime.datetime.now().day,
                "hour": hour,
                "minute": 0,
                "day_of_week": day_of_week,
                "is_peak_hour": is_peak_hour,
                "is_weekend": is_weekend,
                "season": season
            }])

            # -----------------------------
            # ALIGN FEATURES
            # -----------------------------
            for col in columns:
                if col not in df.columns:
                    df[col] = 0

            df = df[columns]

            # -----------------------------
            # ML PREDICTION
            # -----------------------------
            ml_pred = int(model.predict(df)[0])
            proba = model.predict_proba(df)[0]
            confidence = float(max(proba))

            # -----------------------------
            # FINAL HYBRID RESULT
            # -----------------------------
            final_pred = max(ml_pred, real_pred)

            # -----------------------------
            # SMART SCORE
            # -----------------------------
            score = duration_in_traffic * (final_pred + 1)

            results.append({
                "route_id": i,
                "distance_km": round(distance / 1000, 2),
                "duration_min": round(duration / 60, 2),
                "real_duration_min": round(duration_in_traffic / 60, 2),
                "traffic_factor": round(traffic_factor, 2),
                "ml_prediction": ml_pred,
                "real_prediction": real_pred,
                "final_prediction": final_pred,
                "confidence": confidence,
                "score": score
            })

        # -----------------------------
        # BEST ROUTE
        # -----------------------------
        best_route = min(results, key=lambda x: x["score"])

        return {
            "best_route": best_route,
            "all_routes": results
        }

    except HTTPException as he:
        raise he

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))