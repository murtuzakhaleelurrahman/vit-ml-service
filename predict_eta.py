#!/usr/bin/env python3
"""Production ML Inference Script - Predict ETA using Trained XGBoost Model"""

import os
import pickle
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class ETAPredictor:
    """XGBoost-based ETA Predictor"""

    def __init__(self, model_path, features_path):
        print("Loading model artifacts...")

        model_path = os.path.abspath(model_path)
        features_path = os.path.abspath(features_path)

        print(f"Model path: {model_path}")
        print(f"Features path: {features_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Feature file not found: {features_path}")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        print("✓ Model loaded")

        with open(features_path, "rb") as f:
            self.feature_columns = pickle.load(f)

        print("✓ Feature columns loaded")
        print("Predictor ready!\n")

    def predict_speed(
        self,
        hour_of_day,
        is_weekend,
        seg_speed_last_1,
        seg_speed_last_3_mean,
        seg_speed_last_6_mean,
        seg_speed_std_6
    ):
        hour_radians = 2 * np.pi * hour_of_day / 24
        hour_sin = np.sin(hour_radians)
        hour_cos = np.cos(hour_radians)

        features = {
            "is_weekend": is_weekend,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "hour_of_day": hour_of_day,
            "seg_speed_last_1": seg_speed_last_1,
            "seg_speed_last_3_mean": seg_speed_last_3_mean,
            "seg_speed_last_6_mean": seg_speed_last_6_mean,
            "seg_speed_std_6": seg_speed_std_6
        }

        X = pd.DataFrame([features], columns=self.feature_columns)
        predicted_speed = self.model.predict(X)[0]
        return predicted_speed

    def predict_eta(
        self,
        segment_distance_m,
        hour_of_day,
        is_weekend,
        seg_speed_last_1,
        seg_speed_last_3_mean,
        seg_speed_last_6_mean,
        seg_speed_std_6
    ):
        predicted_speed_kmh = self.predict_speed(
            hour_of_day,
            is_weekend,
            seg_speed_last_1,
            seg_speed_last_3_mean,
            seg_speed_last_6_mean,
            seg_speed_std_6
        )

        predicted_speed_mps = predicted_speed_kmh * (1000.0 / 3600.0)
        eta_seconds = segment_distance_m / predicted_speed_mps
        eta_minutes = eta_seconds / 60.0

        return {
            "predicted_speed_kmh": round(float(predicted_speed_kmh), 2),
            "predicted_eta_seconds": round(float(eta_seconds), 2),
            "predicted_eta_minutes": round(float(eta_minutes), 2)
        }
