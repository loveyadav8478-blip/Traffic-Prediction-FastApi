import os
import datetime
import pandas as pd
import joblib
import googlemaps

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
import jwt

# LOAD ENV
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")
JWT_SECRET = os.getenv("JWT_SECRET")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_MAPS_API_KEY")

if not API_SECRET_KEY:
    raise ValueError("Missing API_SECRET_KEY")

if not JWT_SECRET:
    raise ValueError("Missing JWT_SECRET")

# FASTAPI INIT (Docs hidden)
app = FastAPI(
    title="Traffic Route Prediction API 🚗",
    docs_url=None,
    redoc_url=None
)

# LOAD MODEL
model = joblib.load("model_XG.pkl")
columns = joblib.load("columns.pkl")

# GOOGLE MAPS CLIENT
gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

# SECURITY

#API KEY CHECK
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

#JWT CHECK
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
        return payload
    except:
        raise HTTPException(status_code=401, detail="Invalid Token")

#AUTH ROUTES
class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(data: LoginRequest):
    # Dummy user (replace with DB later)
    if data.username == "admin" and data.password == "admin123":
        payload = {
            "sub": data.username,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
        return {"access_token": token}

    raise HTTPException(status_code=401, detail="Invalid credentials")

# INPUT SCHEMA
class RouteInput(BaseModel):
    source: str
    destination: str

# HELPER FUNCTIONS
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

# MAIN ROUTE (PROTECTED 🔒)
@app.post("/predict-route")
def predict_route(
    data: RouteInput,
    api_key: str = Depends(verify_api_key),
    user=Depends(verify_token)
):
    try:
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

            traffic_factor = duration_in_traffic / duration
            real_pred = real_traffic_level(traffic_factor)

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

            for col in columns:
                if col not in df.columns:
                    df[col] = 0

            df = df[columns]

            ml_pred = int(model.predict(df)[0])
            proba = model.predict_proba(df)[0]
            confidence = float(max(proba))

            final_pred = max(ml_pred, real_pred)
            score = duration_in_traffic * (final_pred + 1)

            results.append({
                "route_id": i,
                "distance_km": round(distance / 1000, 2),
                "duration_min": round(duration / 60, 2),
                "real_duration_min": round(duration_in_traffic / 60, 2),
                "traffic_factor": round(traffic_factor, 2),
                "final_prediction": final_pred,
                "confidence": confidence,
                "score": score
            })

        best_route = min(results, key=lambda x: x["score"])

        return {
            "user": user["sub"],
            "best_route": best_route,
            "all_routes": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))