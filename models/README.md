# ML Models Directory

This directory contains all machine learning models for Parkinson's disease detection across three levels.

## Directory Structure

```
models/
├── level1_voice/           # Voice Analysis ML Model (DEPLOYED)
│   ├── voice_model.pkl     # Trained voice detection model
│   ├── scaler.pkl          # Feature scaler
│   └── notes.txt           # Model details and deployment info
│
├── level2_facial/          # Facial Movement ML Model (DEPLOYED)
│   ├── mediapipe_model/    # MediaPipe face mesh model files
│   ├── facial_classifier.h5 # Facial asymmetry classifier
│   └── notes.txt           # Model details and deployment info
│
└── level3_tremor/          # Tremor Detection ML Model (MOCK - Awaiting Real Model)
    ├── tremor_model.pkl    # <-- Place your trained tremor model here
    ├── feature_scaler.pkl  # <-- Place feature scaler here
    └── README.md           # Instructions for model integration
```

## Current Status

### ✅ Level 1 - Voice Analysis (REAL MODEL)
- **Status**: Deployed and Active
- **Endpoint**: `https://neurovoice-level1-ml.onrender.com/predict`
- **Location**: External API (not in this folder)
- **Features**: Analyzes voice recordings + clinical features (age category, neurological test history, hypertension, UPDRS)
- **Output**: Risk score (0-1) and risk level classification

### ✅ Level 2 - Facial Movement (REAL MODEL)
- **Status**: Deployed and Active  
- **Endpoint**: `https://level2-mediapipe-website.onrender.com`
- **Location**: External WebView API (not in this folder)
- **Features**: Real-time facial landmark tracking, blink rate, facial asymmetry, motion analysis
- **Output**: Percentage score, risk level, blink rate, motion metrics, asymmetry detection

### ⏳ Level 3 - Tremor Detection (MOCK MODEL)
- **Status**: Using Mock Predictions (Awaiting Real Model)
- **File**: `backend/inference.py` contains mock implementation
- **Expected Location**: `models/level3_tremor/`
- **Required Files**:
  - `tremor_model.pkl` - Your trained scikit-learn/TensorFlow model
  - `feature_scaler.pkl` - Feature normalization scaler
  - `model_config.json` - Model metadata and feature names

## How to Integrate Your Tremor Model

### Option 1: Scikit-learn Model
```python
import joblib

# Save your trained model
joblib.dump(your_model, 'models/level3_tremor/tremor_model.pkl')
joblib.dump(your_scaler, 'models/level3_tremor/feature_scaler.pkl')
```

### Option 2: TensorFlow/Keras Model
```python
import tensorflow as tf

# Save your model
model.save('models/level3_tremor/tremor_model.h5')
```

### Option 3: PyTorch Model
```python
import torch

# Save your model
torch.save(model.state_dict(), 'models/level3_tremor/tremor_model.pth')
```

## Data Flow

### Voice Test (Level 1)
1. Flutter app → External ML API (https://neurovoice-level1-ml.onrender.com/predict)
2. API returns risk_score, risk_level
3. Flutter app → Backend POST `/api/voice/result` (stores result)
4. Dashboards fetch aggregated voice test history

### Facial Test (Level 2)
1. Flutter WebView → External ML API (https://level2-mediapipe-website.onrender.com)
2. JavaScript channel receives percentage, level, facial metrics
3. Flutter app → Backend POST `/api/facial/result` (stores result)
4. Dashboards fetch aggregated facial test history

### Tremor Test (Level 3)
1. Flutter app collects accelerometer/gyroscope sensor data
2. Flutter app → Backend POST `/api/tremor/analyze` (sends sensor data)
3. Backend loads model from `models/level3_tremor/`
4. Backend runs inference and returns predictions
5. Backend stores result automatically
6. Dashboards fetch aggregated tremor test history

## Model Requirements

Your tremor detection model should:
- Accept accelerometer and gyroscope time-series data
- Output risk level classification (Low/Medium/High)
- Provide confidence score (0-1 range)
- Be serialized as `.pkl`, `.h5`, or `.pth` file

## Updating inference.py

Once you place your model files here, update `backend/inference.py`:

```python
import joblib
import numpy as np

# Load your actual model
model = joblib.load('models/level3_tremor/tremor_model.pkl')
scaler = joblib.load('models/level3_tremor/feature_scaler.pkl')

def predict_tremor_risk(sensor_data):
    """Real ML inference using your trained model"""
    
    # Extract features from sensor data
    features = extract_features(sensor_data)
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    confidence = model.predict_proba(features_scaled)[0].max()
    
    return {
        'risk_level': prediction,
        'confidence': confidence,
        # ... other metrics
    }
```

## Dashboard Data Integration

All three levels now feed into unified dashboards:

**Caregiver Dashboard**:
- Speech Stability: Uses REAL voice test results from Level 1
- Emotional Health: Mock data (future integration)
- Medication: Mock reminders (future integration)
- Alerts: Mock data (future integration)

**Doctor Dashboard**:
- Risk Severity: Calculated from REAL Level 1 + Level 2 + Mock Level 3
- Disease Progression: 6-month trends from all REAL test history
- AI Clinical Summary: Generated from actual test results
- Level Assessments: Individual scores from all three levels

## Notes

- Level 1 and 2 models are already deployed and working
- This folder is for LOCAL model files only (not needed for Level 1 & 2)
- Place your tremor model here when ready
- Mock data is only used for Level 3 until real model is provided

