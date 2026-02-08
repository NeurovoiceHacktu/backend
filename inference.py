"""
Tremor Detection ML Model Inference
Using trained Random Forest model with feature extraction
"""

import numpy as np
import random
import joblib
import os
from pathlib import Path
from feature_extraction import extract_all_features

# Load the trained model and scaler
MODEL_DIR = Path(__file__).parent / 'models' / 'level3_tremor'
MODEL_PATH = MODEL_DIR / 'tremor_model.pkl'
SCALER_PATH = MODEL_DIR / 'scaler.pkl'

# Global variables to hold loaded models
_model = None
_scaler = None

def load_model():
    """Load the trained model and scaler (lazy loading)"""
    global _model, _scaler
    
    if _model is None:
        try:
            print(f"🔄 Loading tremor detection model from {MODEL_PATH}")
            _model = joblib.load(MODEL_PATH)
            _scaler = joblib.load(SCALER_PATH)
            print(f"✅ Model loaded successfully: {type(_model).__name__}")
        except FileNotFoundError:
            print(f"⚠️ Model files not found at {MODEL_DIR}")
            print("⚠️ Falling back to mock predictions")
            return None, None
        except Exception as e:
            print(f"⚠️ Error loading model: {str(e)}")
            print("⚠️ Falling back to mock predictions")
            return None, None
    
    return _model, _scaler


def predict_tremor_risk(sensor_data):
    """
    Predict tremor risk from sensor data using trained ML model
    
    Args:
        sensor_data: List of sensor readings with timestamps
                    Format: [{'timestamp': float, 'x': float, 'y': float, 'z': float}, ...]
                    OR [{'timestamp': float, 'ax': float, 'ay': float, 'az': float, ...}, ...]
    
    Returns:
        Dictionary with prediction results
    """
    
    # Load model
    model, scaler = load_model()
    
    # If model not available, use mock predictions
    if model is None or scaler is None:
        return _mock_prediction(sensor_data)
    
    try:
        # Extract accelerometer data from sensor readings
        acc_x, acc_y, acc_z, timestamps = _extract_sensor_arrays(sensor_data)
        
        if len(acc_x) < 10:
            print(f"⚠️ Insufficient data points: {len(acc_x)}. Using mock prediction.")
            return _mock_prediction(sensor_data)
        
        # Calculate magnitude
        magnitude = np.sqrt(np.array(acc_x)**2 + np.array(acc_y)**2 + np.array(acc_z)**2)
        
        # Calculate sampling rate
        if len(timestamps) > 1:
            dt = np.diff(timestamps)
            sampling_rate = 1.0 / np.mean(dt) if np.mean(dt) > 0 else 100.0
        else:
            sampling_rate = 100.0
        
        # Extract features using the feature_extraction module
        features = extract_all_features(magnitude, sampling_rate)
        
        # Convert to array in correct order
        feature_names = ['mean', 'std', 'variance', 'rms', 'max', 'min', 
                         'range', 'skewness', 'kurtosis', 'dominant_frequency', 
                         'spectral_entropy', 'spectral_energy']
        
        feature_vector = np.array([features[name] for name in feature_names]).reshape(1, -1)
        
        # Scale features
        feature_vector_scaled = scaler.transform(feature_vector)
        
        # Predict
        prediction = model.predict(feature_vector_scaled)[0]
        probabilities = model.predict_proba(feature_vector_scaled)[0]
        
        # Map prediction to risk level
        # prediction: 0 = No Tremor, 1 = Tremor Detected
        has_tremor = bool(prediction == 1)
        confidence = float(max(probabilities))
        tremor_probability = float(probabilities[1])
        
        # Determine risk level based on tremor probability
        if tremor_probability >= 0.7:
            risk_level = 'High'
        elif tremor_probability >= 0.4:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        # Calculate severity score (0-100)
        severity_score = int(tremor_probability * 100)
        
        # Estimate tremor frequency from dominant frequency
        tremor_frequency = features.get('dominant_frequency', 5.5)
        # Typical PD tremor: 4-6 Hz, clip to realistic range
        tremor_frequency = np.clip(tremor_frequency, 3.0, 12.0)
        
        result = {
            'risk_level': risk_level,
            'confidence': round(confidence * 100, 2),
            'tremor_detected': has_tremor,
            'tremor_probability': round(tremor_probability * 100, 2),
            'tremor_frequency': round(tremor_frequency, 2),
            'severity_score': severity_score,
            'recommendations': get_recommendations(risk_level),
            'model_used': 'RandomForest (Real ML Model)'
        }
        
        print(f"✅ Real ML Prediction: {risk_level} risk (confidence: {result['confidence']:.1f}%)")
        
        return result
    
    except Exception as e:
        print(f"⚠️ Error in real model prediction: {str(e)}")
        print(f"⚠️ Falling back to mock prediction")
        return _mock_prediction(sensor_data)


def _extract_sensor_arrays(sensor_data):
    """
    Extract accelerometer arrays from sensor data
    
    Returns:
        tuple: (acc_x, acc_y, acc_z, timestamps)
    """
    acc_x = []
    acc_y = []
    acc_z = []
    timestamps = []
    
    for reading in sensor_data:
        if isinstance(reading, dict):
            # Try different key formats
            x = reading.get('x') or reading.get('ax') or reading.get('acc_x', 0)
            y = reading.get('y') or reading.get('ay') or reading.get('acc_y', 0)
            z = reading.get('z') or reading.get('az') or reading.get('acc_z', 0)
            t = reading.get('timestamp') or reading.get('time', len(timestamps) * 0.01)
            
            acc_x.append(float(x))
            acc_y.append(float(y))
            acc_z.append(float(z))
            timestamps.append(float(t))
    
    return acc_x, acc_y, acc_z, timestamps


def _mock_prediction(sensor_data):
    """
    Generate mock prediction when model is not available
    Uses sensor data variance for more realistic varying results
    """
    try:
        # Extract features from sensor data for varied results
        if isinstance(sensor_data, list) and len(sensor_data) > 0:
            if isinstance(sensor_data[0], dict):
                x_values = [d.get('x', 0) or d.get('ax', 0) for d in sensor_data]
                y_values = [d.get('y', 0) or d.get('ay', 0) for d in sensor_data]
                z_values = [d.get('z', 0) or d.get('az', 0) for d in sensor_data]
                
                # Calculate variance for realistic variation
                x_std = np.std(x_values) if x_values else 0
                y_std = np.std(y_values) if y_values else 0
                z_std = np.std(z_values) if z_values else 0
                
                # Base confidence on actual data variance
                total_variance = x_std + y_std + z_std
                base_confidence = min(90, max(50, 55 + (total_variance * 15)))
                
                # Add small random variation for different results each time
                confidence = base_confidence + random.uniform(-5, 5)
            else:
                confidence = random.uniform(55, 85)
        else:
            confidence = random.uniform(55, 85)
        
        # Determine risk level
        if confidence >= 75:
            risk_level = 'High'
        elif confidence >= 60:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        # Tremor frequency varies realistically
        tremor_frequency = random.uniform(4.0, 7.5)  # Typical PD tremor range
        severity_score = int(confidence * 0.85)
        
        result = {
            'risk_level': risk_level,
            'confidence': round(confidence, 2),
            'tremor_detected': confidence > 65,
            'tremor_probability': round(confidence, 2),
            'tremor_frequency': round(tremor_frequency, 2),
            'severity_score': severity_score,
            'recommendations': get_recommendations(risk_level),
            'model_used': 'Mock Prediction (Variance-based)'
        }
        
        print(f"⚠️ Mock Prediction: {risk_level} risk (confidence: {confidence:.1f}%)")
        
        return result
        
    except Exception as e:
        print(f"⚠️ Error in mock prediction: {str(e)}")
        # Fallback to completely random but valid result
        return {
            'risk_level': 'Medium',
            'confidence': 70.0,
            'tremor_detected': True,
            'tremor_probability': 70.0,
            'tremor_frequency': 5.5,
            'severity_score': 65,
            'recommendations': get_recommendations('Medium'),
            'model_used': 'Mock Prediction (Fallback)'
        }


def get_recommendations(risk_level):
    """Get recommendations based on risk level"""
    recommendations = {
        'Low': [
            'Continue regular monitoring',
            'Maintain healthy lifestyle',
            'Practice hand exercises daily'
        ],
        'Medium': [
            'Schedule follow-up with neurologist',
            'Monitor symptoms closely',
            'Ensure medication compliance',
            'Consider adjusting daily activities'
        ],
        'High': [
            'Immediate consultation recommended',
            'Review medication with doctor',
            'Avoid activities requiring fine motor control',
            'Consider increasing support at home'
        ]
    }
    
    return recommendations.get(risk_level, recommendations['Medium'])
