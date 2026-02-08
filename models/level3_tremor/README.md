# Level 3 - Tremor Detection Model

## Current Status: MOCK PREDICTIONS

This folder is where you should place your trained tremor detection model.

## Required Files

```
level3_tremor/
├── tremor_model.pkl       # Your trained model (scikit-learn)
├── feature_scaler.pkl     # Feature normalization scaler
└── model_config.json      # Optional: model metadata
```

## Model Specifications

### Input Data
Your model should accept sensor data from mobile device accelerometer and gyroscope:
- **Accelerometer**: x, y, z acceleration values
- **Gyroscope**: x, y, z rotation values
- **Sampling Rate**: Typically 50-100 Hz
- **Duration**: 10-30 seconds of data per test

### Expected Features
Extract these features from raw sensor data:
- Mean, std, min, max of each axis
- RMS (Root Mean Square)
- Peak-to-peak amplitude
- Dominant frequency (FFT)
- Zero-crossing rate
- Tremor intensity score

### Output Format
Your model must return:
```python
{
    'risk_level': 'Low' | 'Medium' | 'High',
    'confidence': 0.0 - 1.0,  # Model confidence
    'tremor_frequency': 4.0 - 12.0,  # Hz (typical PD tremor: 4-6 Hz)
    'severity_score': 0 - 100,  # Overall severity
    'recommendations': List[str]  # Clinical recommendations
}
```

## Integration Steps

### 1. Train Your Model
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Your training code here
model = RandomForestClassifier(...)
scaler = StandardScaler()

# Train model
X_train_scaled = scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)

# Save model
joblib.dump(model, 'backend/models/level3_tremor/tremor_model.pkl')
joblib.dump(scaler, 'backend/models/level3_tremor/feature_scaler.pkl')
```

### 2. Update backend/inference.py
Replace the mock implementation with real inference:

```python
import joblib
import numpy as np
from pathlib import Path

# Load model at startup
MODEL_PATH = Path(__file__).parent / 'models' / 'level3_tremor' / 'tremor_model.pkl'
SCALER_PATH = Path(__file__).parent / 'models' / 'level3_tremor' / 'feature_scaler.pkl'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def extract_features(sensor_data):
    """Extract ML features from raw sensor data"""
    # Your feature extraction logic
    features = []
    # ... calculate features
    return np.array(features)

def predict_tremor_risk(sensor_data):
    """Real ML inference"""
    # Extract features
    features = extract_features(sensor_data)
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    confidence = model.predict_proba(features_scaled)[0].max()
    
    # Map prediction to risk level
    risk_levels = {0: 'Low', 1: 'Medium', 2: 'High'}
    risk_level = risk_levels.get(prediction, 'Medium')
    
    return {
        'risk_level': risk_level,
        'confidence': float(confidence),
        'tremor_frequency': calculate_dominant_frequency(sensor_data),
        'severity_score': int(confidence * 100),
        'recommendations': generate_recommendations(risk_level)
    }
```

### 3. Test Your Model
```bash
# Start backend
python backend/app.py

# Test endpoint (from another terminal)
curl -X POST http://localhost:5000/api/tremor/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_data": [
      {"timestamp": 0.0, "ax": 0.1, "ay": 0.2, "az": 9.8, "gx": 0.01, "gy": 0.02, "gz": 0.01},
      {"timestamp": 0.02, "ax": 0.15, "ay": 0.18, "az": 9.85, "gx": 0.015, "gy": 0.018, "gz": 0.012}
    ]
  }'
```

## Current Implementation (MOCK)

Until you provide your trained model, the system uses mock predictions:
- File: `backend/inference.py`
- Method: `predict_tremor_risk()`
- Returns: Randomized but realistic-looking predictions
- Purpose: Allow UI/UX development without real model

## Example Models

If you need a starting point, consider these architectures:

**Random Forest Classifier**:
- Fast inference
- Interpretable features
- Good for tabular feature data

**LSTM Neural Network**:
- Handles time-series data directly
- Captures temporal patterns
- Requires TensorFlow/PyTorch

**SVM with RBF Kernel**:
- Effective for small datasets
- Good generalization
- Requires feature engineering

## Data Collection Tips

For best model performance:
1. **Controlled Environment**: Minimal external vibrations
2. **Consistent Instructions**: "Hold phone steady" or "Rest arm on table"
3. **Multiple Positions**: Test different postures (rest, postural, action tremor)
4. **Clinical Validation**: Compare with UPDRS tremor scores
5. **Balanced Dataset**: Equal samples of each risk level

## Support

Questions about integration? Check:
- `INTEGRATION_GUIDE.md` in project root
- `backend/README.md` for API details
- Backend console logs for debugging
