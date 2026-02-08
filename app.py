"""
Flask API server for Parkinson's Tremor Detection
Handles tremor analysis, caregiver dashboard, and doctor dashboard endpoints
"""

import os
import random
from datetime import datetime, timedelta

from flask import Flask, request, jsonify
from flask_cors import CORS

from inference import predict_tremor_risk

app = Flask(__name__)
CORS(app)

# -----------------------------
# In-memory storage (demo only)
# -----------------------------
test_results = {
    "voice": [],
    "facial": [],
    "tremor": []
}


# -----------------------------
# Health Check
# -----------------------------
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "parkinson-detection-api",
        "test_results_count": {
            "voice": len(test_results["voice"]),
            "facial": len(test_results["facial"]),
            "tremor": len(test_results["tremor"]),
        },
    })


# -----------------------------
# Tremor Analysis
# -----------------------------
@app.route("/api/tremor/analyze", methods=["POST"])
def analyze_tremor():
    try:
        data = request.get_json(force=True)
        sensor_data = data.get("sensor_data", [])

        if not sensor_data:
            return jsonify({"error": "No sensor data provided"}), 400

        prediction = predict_tremor_risk(sensor_data)

        response = {
            "risk_level": prediction["risk_level"],
            "confidence": prediction["confidence"],
            "tremor_detected": prediction["tremor_detected"],
            "tremor_frequency": prediction["tremor_frequency"],
            "severity_score": prediction["severity_score"],
            "recommendations": prediction["recommendations"],
            "model_used": prediction["model_used"],
            "timestamp": datetime.utcnow().isoformat(),
        }

        test_results["tremor"].append(response)
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Voice Result (Level 1)
# -----------------------------
@app.route("/api/voice/result", methods=["POST"])
def store_voice_result():
    try:
        data = request.get_json(force=True)

        test_results["voice"].append({
            "timestamp": data.get("timestamp", datetime.utcnow().isoformat()),
            "risk_level": data.get("risk_level", "Medium"),
            "confidence": float(data.get("risk_score", 0.7)),
            "raw": data,
        })

        return jsonify({"success": True}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Facial Result (Level 2)
# -----------------------------
@app.route("/api/facial/result", methods=["POST"])
def store_facial_result():
    try:
        data = request.get_json(force=True)

        test_results["facial"].append({
            "timestamp": data.get("timestamp", datetime.utcnow().isoformat()),
            "percentage": float(data.get("percentage", 60)),
            "level": data.get("level", "Medium"),
            "raw": data,
        })

        return jsonify({"success": True}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Dashboards (simplified, safe)
# -----------------------------
@app.route("/api/doctor/dashboard", methods=["GET"])
def doctor_dashboard():
    tremor = test_results["tremor"][-1] if test_results["tremor"] else None

    return jsonify({
        "risk_severity": tremor or {
            "risk_level": "Medium",
            "severity_score": 65
        },
        "last_updated": datetime.utcnow().isoformat()
    })


# -----------------------------
# ENTRY POINT (Render safe)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    print("🚀 Parkinson Detection API starting")
    print(f"📡 Listening on port {port}")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )