#!/usr/bin/env python3
"""
Flask ML Microservice - ETA Prediction API
Production-ready ML inference service for real-time ETA predictions
"""

import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Add parent directory to path to import ETAPredictor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict_eta import ETAPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ============================================================
# GLOBAL MODEL INITIALIZATION (Load once at startup)
# ============================================================

logger.info("Initializing ETA Predictor...")

# Get paths to model files (in parent directory)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'xgb_eta_model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'feature_columns.pkl')

# Load predictor once at startup
try:
    predictor = ETAPredictor(model_path=MODEL_PATH, features_path=FEATURES_PATH)
    logger.info("✓ ETA Predictor loaded successfully")
except Exception as e:
    logger.error(f"✗ Failed to load predictor: {str(e)}")
    predictor = None

# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    Returns service status and model availability.
    """
    if predictor is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Model not loaded',
            'service': 'ml-eta-prediction'
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'message': 'Service is running',
        'service': 'ml-eta-prediction',
        'model_loaded': True
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict ETA for a segment.
    
    Request Body (JSON):
    {
        "segment_distance_m": float,     # Segment distance in meters
        "hour_of_day": int,              # Hour (0-23)
        "is_weekend": int,               # 0 or 1
        "seg_speed_last_1": float,       # Last speed (km/h)
        "seg_speed_last_3_mean": float,  # Avg of last 3 speeds (km/h)
        "seg_speed_last_6_mean": float,  # Avg of last 6 speeds (km/h)
        "seg_speed_std_6": float         # Std dev of last 6 speeds (km/h)
    }
    
    Response (JSON):
    {
        "predicted_speed_kmh": float,
        "predicted_eta_seconds": float,
        "predicted_eta_minutes": float
    }
    """
    # Check if model is loaded
    if predictor is None:
        logger.error("Prediction attempted but model not loaded")
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Service is unavailable'
        }), 503
    
    # Validate request
    if not request.is_json:
        logger.warning("Non-JSON request received")
        return jsonify({
            'error': 'Invalid request',
            'message': 'Request must be JSON'
        }), 400
    
    data = request.get_json()
    
    # Validate required fields
    required_fields = [
        'segment_distance_m',
        'hour_of_day',
        'is_weekend',
        'seg_speed_last_1',
        'seg_speed_last_3_mean',
        'seg_speed_last_6_mean',
        'seg_speed_std_6'
    ]
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        logger.warning(f"Missing fields in request: {missing_fields}")
        return jsonify({
            'error': 'Missing required fields',
            'missing_fields': missing_fields,
            'required_fields': required_fields
        }), 400
    
    # Extract and validate input values
    try:
        segment_distance_m = float(data['segment_distance_m'])
        hour_of_day = int(data['hour_of_day'])
        is_weekend = int(data['is_weekend'])
        seg_speed_last_1 = float(data['seg_speed_last_1'])
        seg_speed_last_3_mean = float(data['seg_speed_last_3_mean'])
        seg_speed_last_6_mean = float(data['seg_speed_last_6_mean'])
        seg_speed_std_6 = float(data['seg_speed_std_6'])
        
        # Validate ranges
        if not (0 <= hour_of_day <= 23):
            raise ValueError("hour_of_day must be between 0 and 23")
        if is_weekend not in [0, 1]:
            raise ValueError("is_weekend must be 0 or 1")
        if segment_distance_m <= 0:
            raise ValueError("segment_distance_m must be positive")
        if seg_speed_last_1 < 0 or seg_speed_last_3_mean < 0 or seg_speed_last_6_mean < 0:
            raise ValueError("Speed values must be non-negative")
        if seg_speed_std_6 < 0:
            raise ValueError("seg_speed_std_6 must be non-negative")
            
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid input values: {str(e)}")
        return jsonify({
            'error': 'Invalid input values',
            'message': str(e)
        }), 400
    
    # Make prediction
    try:
        result = predictor.predict_eta(
            segment_distance_m=segment_distance_m,
            hour_of_day=hour_of_day,
            is_weekend=is_weekend,
            seg_speed_last_1=seg_speed_last_1,
            seg_speed_last_3_mean=seg_speed_last_3_mean,
            seg_speed_last_6_mean=seg_speed_last_6_mean,
            seg_speed_std_6=seg_speed_std_6
        )
        
        logger.info(f"Prediction successful: {result['predicted_eta_seconds']:.2f}s")
        
        # Return only essential fields (convert to native Python types for JSON)
        response = {
            'predicted_speed_kmh': float(result['predicted_speed_kmh']),
            'predicted_eta_seconds': float(result['predicted_eta_seconds']),
            'predicted_eta_minutes': float(result['predicted_eta_minutes'])
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/', methods=['GET'])
def index():
    """
    Service information endpoint.
    """
    return jsonify({
        'service': 'ML ETA Prediction Service',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'health': 'GET /health',
            'predict': 'POST /predict'
        },
        'documentation': {
            'predict_example': {
                'url': 'POST http://localhost:5001/predict',
                'body': {
                    'segment_distance_m': 1500,
                    'hour_of_day': 17,
                    'is_weekend': 0,
                    'seg_speed_last_1': 22.0,
                    'seg_speed_last_3_mean': 21.5,
                    'seg_speed_last_6_mean': 20.8,
                    'seg_speed_std_6': 3.5
                }
            }
        }
    }), 200


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not found',
        'message': 'The requested endpoint does not exist'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print('\n' + '='*70)
    print('ML ETA PREDICTION SERVICE')
    print('='*70)
    print('\n🚀 Starting Flask server on port 5001...\n')
    print('Endpoints:')
    print('   GET  http://localhost:5001/         - Service info')
    print('   GET  http://localhost:5001/health   - Health check')
    print('   POST http://localhost:5001/predict  - Make prediction\n')
    print('='*70)
    print('\n📝 EXAMPLE CURL TESTS:\n')
    print('1. Health Check:')
    print('   curl http://localhost:5001/health\n')
    print('2. Predict ETA (Windows PowerShell):')
    print('   $body = @{')
    print('       segment_distance_m = 1500')
    print('       hour_of_day = 17')
    print('       is_weekend = 0')
    print('       seg_speed_last_1 = 22.0')
    print('       seg_speed_last_3_mean = 21.5')
    print('       seg_speed_last_6_mean = 20.8')
    print('       seg_speed_std_6 = 3.5')
    print('   } | ConvertTo-Json')
    print('   Invoke-RestMethod -Uri http://localhost:5001/predict -Method POST -Body $body -ContentType "application/json"\n')
    print('3. Predict ETA (Linux/Mac/Git Bash):')
    print('   curl -X POST http://localhost:5001/predict \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"segment_distance_m":1500,"hour_of_day":17,"is_weekend":0,"seg_speed_last_1":22.0,"seg_speed_last_3_mean":21.5,"seg_speed_last_6_mean":20.8,"seg_speed_std_6":3.5}\'\n')
    print('='*70 + '\n')
    
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
