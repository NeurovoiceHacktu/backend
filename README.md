# Parkinson's Tremor Detection Backend

Python Flask API server for tremor analysis, caregiver dashboard, and doctor dashboard.

## Setup

### 1. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Add Your ML Model (IMPORTANT!)

Currently using mock predictions. To integrate your actual model:

1. **Copy your model file** to `backend/models/` folder:
   ```
   backend/models/tremor_model.pkl
   ```

2. **Update `inference.py`**:
   - Uncomment the joblib import
   - Load your model at the top
   - Update `predict_tremor_risk()` function with your prediction logic
   - Update `extract_features()` with your feature engineering

3. **Provide these details**:
   - Model input format (what features does it expect?)
   - Model output format (what does it return?)
   - Any preprocessing requirements

## Running the Server

```bash
python app.py
```

Server will start on `http://localhost:5000`

## API Endpoints

### 1. Tremor Analysis
```
POST /api/tremor/analyze
Content-Type: application/json

{
  "sensor_data": [
    {"timestamp": 0, "x": 0.1, "y": 0.2, "z": 0.3},
    {"timestamp": 0.01, "x": 0.15, "y": 0.22, "z": 0.31},
    ...
  ]
}

Response:
{
  "risk_level": "High|Medium|Low",
  "confidence": 85.5,
  "tremor_frequency": 6.2,
  "severity_score": 78,
  "recommendations": [...]
}
```

### 2. Caregiver Dashboard
```
GET /api/caregiver/dashboard

Response:
{
  "patient_name": "Arthur Morgan",
  "emotional_health": {...},
  "medication": {...},
  "speech_stability": {...},
  "emergency_alerts": [...]
}
```

### 3. Doctor Dashboard
```
GET /api/doctor/dashboard

Response:
{
  "patient_name": "Arthur Morgan",
  "risk_severity": {...},
  "ai_clinical_summary": {...},
  "disease_progression": {...}
}
```

### 4. Patient History
```
GET /api/patient/history

Response:
{
  "history": [...]
}
```

## Fake Data

All dashboard endpoints generate fresh fake data on each request to simulate real-time updates. This makes the app feel dynamic during development.

## Next Steps

1. ✅ Backend structure created
2. ⏳ Add your actual ML model
3. ⏳ Connect Flutter app to this API
4. ⏳ Test with real sensor data
5. ⏳ Deploy to production server

## Notes

- CORS is enabled for all origins (tighten in production)
- In-memory storage (use database for production)
- Debug mode enabled (disable in production)
