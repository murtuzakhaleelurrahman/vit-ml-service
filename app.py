#!/usr/bin/env python3
"""
Flask ML Microservice - ETA Prediction API
Production-ready ML inference service for real-time ETA predictions
"""

import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from predict_eta import ETAPredictor

# ------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Flask App Initialization
# ------------------------------------------------------------

app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------
# Model Initialization
# ------------------------------------------------------------

logger.info("Initializing ETA Predictor...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgb_eta_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")

predictor = None
load_error = None

try:
    predictor = ETAPredictor(
        model_path=MODEL_PATH,
        features_path=FEATURES_PATH
    )
    logger.info("✓ ETA Predictor loaded successfully")
except Exception as e:
    load_error = str(e)
    logger.error(f"✗ Failed to load predictor: {load_error}")
    predictor = None

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    if predictor is None:
        return jsonify({
            "status": "unhealthy",
            "message": "Model not loaded",
            "error": load_error
        }), 503

    return jsonify({
        "status": "healthy",
        "service": "ml-eta-prediction",
        "model_loaded": True
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    if predictor is None:
        return jsonify({
            "error": "Model not loaded",
            "message": load_error
        }), 503

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    required_fields = [
        "segment_distance_m",
        "hour_of_day",
        "is_weekend",
        "seg_speed_last_1",
        "seg_speed_last_3_mean",
        "seg_speed_last_6_mean",
        "seg_speed_std_6"
    ]

    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": "Missing fields", "missing": missing}), 400

    try:
        result = predictor.predict_eta(
            segment_distance_m=float(data["segment_distance_m"]),
            hour_of_day=int(data["hour_of_day"]),
            is_weekend=int(data["is_weekend"]),
            seg_speed_last_1=float(data["seg_speed_last_1"]),
            seg_speed_last_3_mean=float(data["seg_speed_last_3_mean"]),
            seg_speed_last_6_mean=float(data["seg_speed_last_6_mean"]),
            seg_speed_std_6=float(data["seg_speed_std_6"])
        )
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "ML ETA Prediction Service",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict"
        }
    }), 200


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
