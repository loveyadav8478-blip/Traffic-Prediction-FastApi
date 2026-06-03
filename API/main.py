import os
import logging
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
from typing import Optional

from auth import (
    create_token, verify_api_key, verify_token,
    ADMIN_USERNAME, ADMIN_PASSWORD
)
from predictor import predict_traffic

# ── Startup ────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_required = {"API_SECRET_KEY", "JWT_SECRET", "ADMIN_PASSWORD"}
_missing  = [k for k in _required if not os.getenv(k)]
if _missing:
    raise RuntimeError(f"❌ Missing required env vars: {_missing}")

app = FastAPI(
    title="Routex Traffic Prediction API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str


class RouteInput(BaseModel):
    source:      str
    destination: str
    weather:     Optional[str]   = "Clear"
    holiday:     Optional[str]   = "No Holiday"
    temp_cel:    Optional[float] = 28.0
    hour:        Optional[int]   = None   # 0-23, parsed from "HH:mm" in Spring Boot
    day_of_week: Optional[int]   = None   # 0=Mon … 6=Sun
    distance_km: Optional[float] = None   # optional — auto-fetched if not provided

    @field_validator("source", "destination")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


# ── Endpoints ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":  "ok",
        "service": "routex-traffic-api",
        "model":   "india_v2",
        "version": "2.0.0",
    }


@app.post("/login")
def login(data: LoginRequest):
    if data.username != ADMIN_USERNAME or data.password != ADMIN_PASSWORD:
        logger.warning("Failed login attempt for user: %s", data.username)
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(data.username)
    return {"access_token": token, "token_type": "Bearer"}


@app.post("/predict-route")
def predict_route(
    data:    RouteInput,
    api_key: str  = Depends(verify_api_key),
    user:    dict = Depends(verify_token),
):
    logger.info("Prediction | user=%s | %s → %s",
                user.get("sub"), data.source, data.destination)

    result = predict_traffic(
        source      = data.source,
        destination = data.destination,
        hour        = data.hour,
        day_of_week = data.day_of_week,
        weather     = data.weather,
        holiday     = data.holiday,
        distance_km = data.distance_km,
        temp_cel    = data.temp_cel,
    )

    return {
        "user":        user.get("sub"),
        "source":      data.source,
        "destination": data.destination,
        "best_route":  result,
    }