#!/usr/bin/env python3
"""Production ML Inference Script - Predict ETA using Trained XGBoost Model"""

import pickle
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

class ETAPredictor:
    """XGBoost-based ETA Predictor"""
    
    def __init__(self, model_path='xgb_eta_model.pkl', features_path='feature_columns.pkl'):
        """Initialize predictor by loading trained model and feature columns"""
        print('Loading model artifacts...')
        
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print('   ✓ Loaded: ' + model_path)
        
        # Load feature columns
        with open(features_path, 'rb') as f:
            self.feature_columns = pickle.load(f)
        print('   ✓ Loaded: ' + features_path)
        
        print('Predictor ready!\n')
    
    def predict_speed(self, hour_of_day, is_weekend, seg_speed_last_1, 
                      seg_speed_last_3_mean, seg_speed_last_6_mean, seg_speed_std_6):
        """
        Predict segment speed given input features.
        
        Parameters:
        -----------
        hour_of_day : int
            Hour of day (0-23)
        is_weekend : int
            1 if weekend, 0 if weekday
        seg_speed_last_1 : float
            Last observed speed on this segment (km/h)
        seg_speed_last_3_mean : float
            Mean of last 3 speeds on this segment (km/h)
        seg_speed_last_6_mean : float
            Mean of last 6 speeds on this segment (km/h)
        seg_speed_std_6 : float
            Standard deviation of last 6 speeds on this segment (km/h)
        
        Returns:
        --------
        float
            Predicted speed (km/h)
        """
        # Compute cyclic time features
        hour_radians = 2 * np.pi * hour_of_day / 24
        hour_sin = np.sin(hour_radians)
        hour_cos = np.cos(hour_radians)
        
        # Create feature dictionary
        features = {
            'is_weekend': is_weekend,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'hour_of_day': hour_of_day,
            'seg_speed_last_1': seg_speed_last_1,
            'seg_speed_last_3_mean': seg_speed_last_3_mean,
            'seg_speed_last_6_mean': seg_speed_last_6_mean,
            'seg_speed_std_6': seg_speed_std_6
        }
        
        # Create DataFrame with correct column order
        X = pd.DataFrame([features], columns=self.feature_columns)
        
        # Predict
        predicted_speed = self.model.predict(X)[0]
        
        return predicted_speed
    
    def predict_eta(self, segment_distance_m, hour_of_day, is_weekend, 
                    seg_speed_last_1, seg_speed_last_3_mean, 
                    seg_speed_last_6_mean, seg_speed_std_6):
        """
        Predict ETA in seconds for a segment given distance and features.
        
        Parameters:
        -----------
        segment_distance_m : float
            Segment distance in meters
        hour_of_day : int
            Hour of day (0-23)
        is_weekend : int
            1 if weekend, 0 if weekday
        seg_speed_last_1 : float
            Last observed speed on this segment (km/h)
        seg_speed_last_3_mean : float
            Mean of last 3 speeds on this segment (km/h)
        seg_speed_last_6_mean : float
            Mean of last 6 speeds on this segment (km/h)
        seg_speed_std_6 : float
            Standard deviation of last 6 speeds on this segment (km/h)
        
        Returns:
        --------
        dict
            {
                'predicted_speed_kmh': float,
                'predicted_speed_mps': float,
                'predicted_eta_seconds': float,
                'predicted_eta_minutes': float
            }
        """
        # Predict speed
        predicted_speed_kmh = self.predict_speed(
            hour_of_day, is_weekend, seg_speed_last_1,
            seg_speed_last_3_mean, seg_speed_last_6_mean, seg_speed_std_6
        )
        
        # Convert to m/s
        predicted_speed_mps = predicted_speed_kmh * (1000.0 / 3600.0)
        
        # Compute ETA
        eta_seconds = segment_distance_m / predicted_speed_mps
        eta_minutes = eta_seconds / 60.0
        
        return {
            'predicted_speed_kmh': round(predicted_speed_kmh, 2),
            'predicted_speed_mps': round(predicted_speed_mps, 2),
            'predicted_eta_seconds': round(eta_seconds, 2),
            'predicted_eta_minutes': round(eta_minutes, 2)
        }


# ============================================================
# EXAMPLE USAGE
# ============================================================

def example_usage():
    """Demonstrate how to use the ETAPredictor"""
    print('='*70)
    print('ETA PREDICTOR - EXAMPLE USAGE')
    print('='*70 + '\n')
    
    # Initialize predictor
    predictor = ETAPredictor(
        model_path='xgb_eta_model.pkl',
        features_path='feature_columns.pkl'
    )
    
    # Example 1: Predict speed only
    print('EXAMPLE 1: Predict Speed')
    print('-' * 70)
    predicted_speed = predictor.predict_speed(
        hour_of_day=9,           # 9 AM
        is_weekend=0,            # Weekday
        seg_speed_last_1=28.0,   # Last observed speed: 28 km/h
        seg_speed_last_3_mean=27.5,  # Mean of last 3: 27.5 km/h
        seg_speed_last_6_mean=26.8,  # Mean of last 6: 26.8 km/h
        seg_speed_std_6=2.1      # Std dev of last 6: 2.1 km/h
    )
    print('Input:')
    print('   Hour: 9 AM (weekday)')
    print('   Last speed: 28.0 km/h')
    print('   Last 3 avg: 27.5 km/h')
    print('   Last 6 avg: 26.8 km/h')
    print('   Last 6 std: 2.1 km/h')
    print('\nOutput:')
    print('   Predicted Speed: ' + str(round(predicted_speed, 2)) + ' km/h\n')
    
    # Example 2: Predict ETA for a segment
    print('EXAMPLE 2: Predict ETA for Segment')
    print('-' * 70)
    eta_result = predictor.predict_eta(
        segment_distance_m=1500,  # 1.5 km segment
        hour_of_day=17,          # 5 PM (peak hour)
        is_weekend=0,            # Weekday
        seg_speed_last_1=22.0,   # Last observed speed: 22 km/h
        seg_speed_last_3_mean=21.5,  # Mean of last 3: 21.5 km/h
        seg_speed_last_6_mean=20.8,  # Mean of last 6: 20.8 km/h
        seg_speed_std_6=3.5      # Std dev of last 6: 3.5 km/h (high variability)
    )
    print('Input:')
    print('   Segment distance: 1500 meters (1.5 km)')
    print('   Hour: 5 PM (weekday, peak hour)')
    print('   Last speed: 22.0 km/h')
    print('   Last 3 avg: 21.5 km/h')
    print('   Last 6 avg: 20.8 km/h')
    print('   Last 6 std: 3.5 km/h')
    print('\nOutput:')
    print('   Predicted Speed: ' + str(eta_result['predicted_speed_kmh']) + ' km/h')
    print('   Predicted Speed: ' + str(eta_result['predicted_speed_mps']) + ' m/s')
    print('   Predicted ETA: ' + str(eta_result['predicted_eta_seconds']) + ' seconds')
    print('   Predicted ETA: ' + str(eta_result['predicted_eta_minutes']) + ' minutes\n')
    
    # Example 3: Weekend vs Weekday comparison
    print('EXAMPLE 3: Weekend vs Weekday Comparison')
    print('-' * 70)
    
    # Same conditions, weekday
    weekday_eta = predictor.predict_eta(
        segment_distance_m=2000,
        hour_of_day=10,
        is_weekend=0,  # Weekday
        seg_speed_last_1=30.0,
        seg_speed_last_3_mean=29.5,
        seg_speed_last_6_mean=29.0,
        seg_speed_std_6=1.5
    )
    
    # Same conditions, weekend
    weekend_eta = predictor.predict_eta(
        segment_distance_m=2000,
        hour_of_day=10,
        is_weekend=1,  # Weekend
        seg_speed_last_1=30.0,
        seg_speed_last_3_mean=29.5,
        seg_speed_last_6_mean=29.0,
        seg_speed_std_6=1.5
    )
    
    print('Weekday (10 AM):')
    print('   ETA: ' + str(weekday_eta['predicted_eta_seconds']) + ' seconds (' + 
          str(weekday_eta['predicted_eta_minutes']) + ' min)')
    
    print('\nWeekend (10 AM):')
    print('   ETA: ' + str(weekend_eta['predicted_eta_seconds']) + ' seconds (' + 
          str(weekend_eta['predicted_eta_minutes']) + ' min)')
    
    eta_diff = weekday_eta['predicted_eta_seconds'] - weekend_eta['predicted_eta_seconds']
    print('\nDifference: ' + str(round(eta_diff, 2)) + ' seconds')
    
    if abs(eta_diff) < 5:
        print('(Minimal difference - traffic similar on weekends)\n')
    elif eta_diff < 0:
        print('(Weekday is faster)\n')
    else:
        print('(Weekend is faster)\n')
    
    print('='*70)
    print('READY FOR PRODUCTION USE')
    print('='*70)
    print('\nIntegration example:')
    print('```python')
    print('from predict_eta import ETAPredictor')
    print('')
    print('# Initialize once')
    print('predictor = ETAPredictor()')
    print('')
    print('# Make predictions')
    print('result = predictor.predict_eta(')
    print('    segment_distance_m=1500,')
    print('    hour_of_day=current_hour,')
    print('    is_weekend=is_weekend_flag,')
    print('    seg_speed_last_1=last_speed,')
    print('    seg_speed_last_3_mean=avg_3,')
    print('    seg_speed_last_6_mean=avg_6,')
    print('    seg_speed_std_6=std_6')
    print(')')
    print('eta = result["predicted_eta_seconds"]')
    print('```\n')


if __name__ == '__main__':
    example_usage()
