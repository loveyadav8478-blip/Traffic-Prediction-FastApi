import os
import logging
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
import datetime
from fastapi import HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

from auth import (
    create_token, verify_api_key, verify_token,
    ADMIN_USERNAME, ADMIN_PASSWORD
)
from predictor import predict

#Startup
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_required = {"API_SECRET_KEY", "JWT_SECRET", "ADMIN_PASSWORD"}
_missing = [k for k in _required if not os.getenv(k)]
if _missing:
    raise RuntimeError(f"❌ Missing required env vars: {_missing}")

app = FastAPI(
    title="Traffic Prediction API",
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#Schemas

class LoginRequest(BaseModel):
    username: str
    password: str


# models.py or inside main.py
class RouteInput(BaseModel):
    source      : str
    destination : str
    weather     : str   = "Clouds"   # optional — default Clouds
    holiday     : str   = "No Holiday"
    temp_cel    : float = 25.0

    @field_validator("source", "destination")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


#Endpoints

@app.get("/health")
def health():
    return {"status": "ok", "service": "traffic-prediction-api"}


@app.post("/login")
def login(data: LoginRequest):
    if data.username != ADMIN_USERNAME or data.password != ADMIN_PASSWORD:
        logger.warning("Failed login attempt for user: %s", data.username)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(data.username)
    return {"access_token": token, "token_type": "Bearer"}


@app.post("/predict-route")
def predict_route(
    data     : RouteInput,
    api_key  : str  = Depends(verify_api_key),
    user     : dict = Depends(verify_token)
):
    logger.info("Prediction | user=%s | %s → %s",
                user.get("sub"), data.source, data.destination)

    result = predict(
        source      = data.source,
        destination = data.destination,
        weather     = data.weather,
        holiday     = data.holiday,
        temp_cel    = data.temp_cel
    )

    return {
        "user"        : user.get("sub"),
        "source"      : data.source,
        "destination" : data.destination,
        "best_route"  : result
    }