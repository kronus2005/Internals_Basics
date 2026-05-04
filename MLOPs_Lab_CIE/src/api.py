"""
Task 3: FastAPI serving.
Start the server:  python src/api.py
Then in a NEW terminal, run:  python src/test_api.py
"""

import os
import json
import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── load model ─────────────────────────────────────────────────────────────────
# Prefer the tuned model if it exists, fall back to base RandomForest
tuned = os.path.join(MODELS_DIR, "best_tuned_model.pkl")
base_rf = os.path.join(MODELS_DIR, "RandomForest.pkl")
base_lr = os.path.join(MODELS_DIR, "LinearRegression.pkl")

if os.path.exists(tuned):
    MODEL_PATH = tuned
elif os.path.exists(base_rf):
    MODEL_PATH = base_rf
else:
    MODEL_PATH = base_lr

model = joblib.load(MODEL_PATH)
print(f"Loaded model: {MODEL_PATH}")


# ── pydantic schema ────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    orbit_altitude_km: float = Field(..., ge=200, le=36000,
        description="Orbit altitude in km (200–36000)")
    ground_station_count: int = Field(..., ge=1, le=10,
        description="Number of ground stations (1–10)")
    atmospheric_index: float = Field(..., ge=1.0, le=5.0,
        description="Atmospheric index (1–5)")
    is_polar_orbit: int = Field(..., ge=0, le=1,
        description="1 if polar orbit, 0 otherwise")


class PredictResponse(BaseModel):
    prediction: float
    unit: str = "milliseconds"


# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="OrbitCalc Signal Delay API", version="1.0")


@app.get("/heartbeat")
def heartbeat():
    return {"alive": True, "service": "OrbitCalc signal_delay_ms API"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    features = np.array([[
        req.orbit_altitude_km,
        req.ground_station_count,
        req.atmospheric_index,
        req.is_polar_orbit,
    ]])
    pred = float(model.predict(features)[0])
    return PredictResponse(prediction=round(pred, 4))


# ── save result JSON (done once at startup, before server blocks) ──────────────
def save_result_json():
    """Call the model once with the test input and save step3_s4.json."""
    test_input = {
        "orbit_altitude_km": 14411.4,
        "ground_station_count": 6,
        "atmospheric_index": 3.0,
        "is_polar_orbit": 0,
    }
    features = np.array([[
        test_input["orbit_altitude_km"],
        test_input["ground_station_count"],
        test_input["atmospheric_index"],
        test_input["is_polar_orbit"],
    ]])
    pred = float(model.predict(features)[0])

    output = {
        "health_endpoint": "/heartbeat",
        "predict_endpoint": "/predict",
        "port": 8080,
        "health_response": {"alive": True, "service": "OrbitCalc signal_delay_ms API"},
        "test_input": test_input,
        "prediction": round(pred, 4),
    }
    out_path = os.path.join(RESULTS_DIR, "step3_s4.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}  (prediction={pred:.4f})")


if __name__ == "__main__":
    save_result_json()          # write JSON before blocking
    uvicorn.run(app, host="0.0.0.0", port=8080)
