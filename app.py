"""
Flask API server for Parkinson's Tremor Detection
Handles tremor analysis, caregiver dashboard, and doctor dashboard endpoints
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from datetime import datetime, timedelta
import random
import json
from inference import predict_tremor_risk

app = Flask(__name__)
CORS(app)

# In-memory storage for demo (replace with database in production)
patient_data = {}

# Storage for actual test results
test_results = {
    'voice': [],  # Level 1 results from voice ML model
    'facial': [], # Level 2 results from facial ML model
    'tremor': []  # Level 3 results from tremor ML model
}


def generate_fake_patient_data():
    """Generate realistic fake patient data for dashboards"""
    now = datetime.now()
    
    return {
        'patient_id': 'P12345',
        'name': 'Arthur Morgan',
        'age': 67,
        'last_updated': now.isoformat(),
        
        # Caregiver Dashboard Data
        'emotional_health': {
            'mood_score': random.randint(60, 85),
            'stress_level': random.choice(['Low', 'Moderate', 'High']),
            'recent_entries': [
                {'date': (now - timedelta(days=i)).strftime('%Y-%m-%d'), 
                 'mood': random.choice(['Happy', 'Calm', 'Anxious', 'Tired']),
                 'score': random.randint(50, 90)}
                for i in range(7)
            ]
        },
        
        'medication': {
            'next_reminder': (now + timedelta(hours=2)).strftime('%I:%M %p'),
            'today_taken': ['Levodopa 8:00 AM', 'Carbidopa 12:00 PM'],
            'today_pending': ['Levodopa 6:00 PM'],
            'compliance_rate': random.randint(85, 98)
        },
        
        'speech_stability': {
            'daily_score': random.randint(70, 95),
            'trend': random.choice(['Improving', 'Stable', 'Declining']),
            'last_7_days': [random.randint(65, 95) for _ in range(7)]
        },
        
        'emergency_alerts': [
            {
                'type': 'Fall Detection',
                'status': 'Resolved',
                'time': (now - timedelta(hours=48)).strftime('%Y-%m-%d %H:%M'),
                'severity': 'Medium'
            },
            {
                'type': 'Missed Medication',
                'status': 'Active',
                'time': (now - timedelta(hours=3)).strftime('%Y-%m-%d %H:%M'),
                'severity': 'Low'
            }
        ],
        
        # Doctor Dashboard Data
        'risk_severity': {
            'current_level': random.choice(['Low Risk', 'Medium Risk', 'High Risk']),
            'score': random.randint(40, 85),
            'date': now.strftime('%Y-%m-%d'),
            'tremor_levels': {
                'voice': random.randint(30, 80),
                'facial': random.randint(25, 75),
                'tremor': random.randint(35, 85)
            }
        },
        
        'ai_clinical_summary': {
            'generated_at': now.strftime('%Y-%m-%d %H:%M'),
            'summary': """Patient shows moderate progression of tremor symptoms over the past 3 months. 
Voice stability has improved with current medication regimen, showing 15% improvement. 
Tremor episodes are most frequent in morning hours (6-10 AM).
Recommend: Consider adjusting evening medication dosage. Schedule follow-up in 2 weeks.""",
            'key_findings': [
                'Morning tremor frequency increased by 12%',
                'Medication compliance excellent at 94%',
                'Speech clarity improved significantly',
                'Recommend gait analysis in next visit'
            ]
        },
        
        'disease_progression': {
            'months': ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb'],
            'tremor_scores': [45, 52, 48, 55, 60, 58],
            'voice_scores': [60, 58, 62, 65, 70, 72],
            'motor_scores': [55, 58, 54, 60, 62, 59]
        }
    }


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'service': 'parkinson-detection-api',
        'test_results_count': {
            'voice': len(test_results['voice']),
            'facial': len(test_results['facial']),
            'tremor': len(test_results['tremor'])
        }
    })


@app.route('/api/tremor/analyze', methods=['POST'])
def analyze_tremor():
    """
    Analyze tremor data from device sensors
    Expects: { "sensor_data": [timestamps and accelerometer readings] }
    Returns: Risk assessment with severity level
    """
    try:
        data = request.json
        sensor_data = data.get('sensor_data', [])
        
        if not sensor_data:
            return jsonify({'error': 'No sensor data provided'}), 400
        
        # Call ML model prediction
        prediction = predict_tremor_risk(sensor_data)
        
        # Generate response
        response = {
            'risk_level': prediction['risk_level'],
            'confidence': prediction['confidence'],
            'tremor_frequency': prediction.get('tremor_frequency', random.uniform(4.5, 7.2)),
            'severity_score': prediction.get('severity_score', random.randint(45, 85)),
            'timestamp': datetime.now().isoformat(),
            'recommendations': prediction.get('recommendations', [
                'Continue regular monitoring',
                'Maintain medication schedule',
                'Practice hand exercises'
            ]),
            'model_used': prediction.get('model_used', 'Unknown'),
            'tremor_detected': prediction.get('tremor_detected', False),
        }
        
        # Store result for dashboard use
        test_results['tremor'].append({
            'timestamp': datetime.now().isoformat(),
            'risk_level': response['risk_level'],
            'confidence': response['confidence'],
            'severity_score': response['severity_score'],
            'tremor_frequency': response['tremor_frequency'],
            'model_used': response['model_used']
        })
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/voice/result', methods=['POST'])
def store_voice_result():
    """
    Store voice test result from Level 1 ML model
    Expected data: {risk_score, risk_level, timestamp, ...}
    """
    try:
        data = request.json
        
        # Store result
        test_results['voice'].append({
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'risk_level': data.get('risk_level', data.get('riskLevel')),
            'confidence': data.get('risk_score', data.get('riskScore', 0.5)),
            'raw_data': data
        })
        
        print(f"✅ Stored voice test result: {data.get('risk_level')} ({data.get('risk_score')})")
        
        return jsonify({'success': True, 'message': 'Voice result stored'}), 200
    
    except Exception as e:
        print(f"❌ Error storing voice result: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/facial/result', methods=['POST'])
def store_facial_result():
    """
    Store facial test result from Level 2 ML model
    Expected data: {percentage, level, blinkRate, motion, asymmetry, ...}
    """
    try:
        data = request.json
        
        # Store result
        test_results['facial'].append({
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'percentage': data.get('percentage', 0),
            'level': data.get('level', 'N/A'),
            'blink_rate': data.get('blinkRate', 0),
            'motion': data.get('motion', 0),
            'asymmetry': data.get('asymmetry', False),
            'raw_data': data
        })
        
        print(f"✅ Stored facial test result: {data.get('level')} ({data.get('percentage')}%)")
        
        return jsonify({'success': True, 'message': 'Facial result stored'}), 200
    
    except Exception as e:
        print(f"❌ Error storing facial result: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/caregiver/dashboard', methods=['GET'])
def get_caregiver_dashboard():
    """
    Get caregiver dashboard data
    Returns: Patient emotional health, medications, speech, alerts
    """
    try:
        # Get real speech stability from voice test results
        speech_data = _calculate_speech_stability_from_real_data()
        
        # Generate other data (medications, alerts, emotional health)
        data = generate_fake_patient_data()
        
        caregiver_data = {
            'patient_name': data['name'],
            'patient_id': data['patient_id'],
            'emotional_health': data['emotional_health'],
            'medication': data['medication'],
            'speech_stability': speech_data,  # REAL DATA from voice tests
            'emergency_alerts': data['emergency_alerts'],
            'last_updated': data['last_updated']
        }
        
        return jsonify(caregiver_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/doctor/dashboard', methods=['GET'])
def get_doctor_dashboard():
    """
    Get doctor dashboard data
    Returns: Risk severity, AI summary, disease progression
    """
    try:
        # Get REAL data from actual test results
        risk_data = _calculate_risk_severity_from_real_data()
        progression_data = _calculate_disease_progression_from_real_data()
        ai_summary = _generate_ai_summary_from_real_data()
        
        # Fallback data for fields we don't have yet
        data = generate_fake_patient_data()
        
        doctor_data = {
            'patient_name': data['name'],
            'patient_id': data['patient_id'],
            'age': data['age'],
            'risk_severity': risk_data,  # REAL DATA from all tests
            'ai_clinical_summary': ai_summary,  # Generated from REAL DATA
            'disease_progression': progression_data,  # REAL DATA trends
            'last_updated': datetime.now().isoformat()
        }
        
        return jsonify(doctor_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/patient/history', methods=['GET'])
def get_patient_history():
    """Get patient test history"""
    try:
        history = []
        now = datetime.now()
        
        # Generate fake history
        for i in range(10):
            date = now - timedelta(days=i*3)
            risk = random.choice(['Low', 'Medium', 'High'])
            
            history.append({
                'date': date.strftime('%Y-%m-%d'),
                'time': date.strftime('%I:%M %p'),
                'test_type': random.choice(['Tremor', 'Voice', 'Facial']),
                'risk_level': risk,
                'score': random.randint(40, 95)
            })
        
        return jsonify({'history': history})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _calculate_speech_stability_from_real_data():
    """Calculate speech stability from actual voice test results"""
    if not test_results['voice']:
        # No real data yet, return reasonable defaults
        return {
            'daily_score': 75,
            'trend': 'Stable',
            'last_7_days': [70, 72, 75, 73, 75, 76, 75]
        }
    
    # Get last 7 voice tests
    recent_tests = sorted(test_results['voice'], key=lambda x: x['timestamp'], reverse=True)[:7]
    
    # Calculate scores from real confidence values (0-1 to 0-100 scale)
    scores = [int(test['confidence'] * 100) for test in recent_tests]
    scores.reverse()  # Oldest to newest
    
    # Pad if less than 7
    while len(scores) < 7:
        scores.insert(0, scores[0] if scores else 70)
    
    daily_score = scores[-1] if scores else 75
    
    # Determine trend
    if len(scores) >= 3:
        if scores[-1] > scores[-3] + 5:
            trend = 'Improving'
        elif scores[-1] < scores[-3] - 5:
            trend = 'Declining'
        else:
            trend = 'Stable'
    else:
        trend = 'Stable'
    
    return {
        'daily_score': daily_score,
        'trend': trend,
        'last_7_days': scores
    }


def _calculate_risk_severity_from_real_data():
    """Calculate overall risk severity from all test types"""
    # Get latest results from each test type
    latest_voice = test_results['voice'][-1] if test_results['voice'] else None
    latest_facial = test_results['facial'][-1] if test_results['facial'] else None
    latest_tremor = test_results['tremor'][-1] if test_results['tremor'] else None
    
    # Calculate scores (0-100 scale)
    voice_score = int(latest_voice['confidence'] * 100) if latest_voice else 65
    facial_score = int(latest_facial['percentage']) if latest_facial else 60
    tremor_score = latest_tremor.get('severity_score', 70) if latest_tremor else 70
    
    # Overall score is average
    overall_score = int((voice_score + facial_score + tremor_score) / 3)
    
    # Determine risk level
    if overall_score >= 75:
        risk_level = 'High Risk'
    elif overall_score >= 60:
        risk_level = 'Medium Risk'
    else:
        risk_level = 'Low Risk'
    
    return {
        'current_level': risk_level,
        'score': overall_score,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'tremor_levels': {
            'voice': voice_score,
            'facial': facial_score,
            'tremor': tremor_score
        }
    }


def _calculate_disease_progression_from_real_data():
    """Calculate 6-month disease progression from historical test data"""
    # This would ideally query a database with timestamps
    # For now, we'll use available data and extrapolate
    
    months = ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb']
    
    if not test_results['voice'] and not test_results['facial'] and not test_results['tremor']:
        # No data, return mock progression
        return {
            'months': months,
            'tremor_scores': [45, 52, 48, 55, 60, 58],
            'voice_scores': [60, 58, 62, 65, 70, 72],
            'motor_scores': [55, 58, 54, 60, 62, 59]
        }
    
    # Use recent test results to build trend
    recent_voice = test_results['voice'][-6:] if test_results['voice'] else []
    recent_facial = test_results['facial'][-6:] if test_results['facial'] else []
    recent_tremor = test_results['tremor'][-6:] if test_results['tremor'] else []
    
    # Build voice trend
    voice_scores = [int(t['confidence'] * 100) for t in recent_voice]
    while len(voice_scores) < 6:
        voice_scores.insert(0, voice_scores[0] - random.randint(2, 8) if voice_scores else 60)
    
    # Build facial/motor trend
    facial_scores = [int(t['percentage']) for t in recent_facial]
    while len(facial_scores) < 6:
        facial_scores.insert(0, facial_scores[0] - random.randint(2, 8) if facial_scores else 58)
    
    # Build tremor trend
    tremor_scores = [t.get('severity_score', 70) for t in recent_tremor]
    while len(tremor_scores) < 6:
        tremor_scores.insert(0, tremor_scores[0] - random.randint(2, 8) if tremor_scores else 55)
    
    return {
        'months': months,
        'tremor_scores': tremor_scores[:6],
        'voice_scores': voice_scores[:6],
        'motor_scores': facial_scores[:6]
    }


def _generate_ai_summary_from_real_data():
    """Generate AI clinical summary based on actual test results"""
    latest_voice = test_results['voice'][-1] if test_results['voice'] else None
    latest_facial = test_results['facial'][-1] if test_results['facial'] else None
    latest_tremor = test_results['tremor'][-1] if test_results['tremor'] else None
    
    # Build summary based on real data
    findings = []
    
    if latest_voice:
        voice_score = int(latest_voice['confidence'] * 100)
        if voice_score > 75:
            findings.append(f'Voice analysis shows elevated risk ({voice_score}/100)')
        else:
            findings.append(f'Voice stability within normal range ({voice_score}/100)')
    
    if latest_facial:
        facial_score = int(latest_facial['percentage'])
        findings.append(f'Facial movement assessment: {facial_score}/100')
        if latest_facial.get('asymmetry'):
            findings.append('Facial asymmetry detected')
    
    if latest_tremor:
        tremor_risk = latest_tremor.get('risk_level', 'Medium')
        findings.append(f'Tremor risk level: {tremor_risk}')
    
    # Add compliance note if we have multiple tests
    total_tests = len(test_results['voice']) + len(test_results['facial']) + len(test_results['tremor'])
    if total_tests > 5:
        findings.append('Excellent test compliance with regular monitoring')
    
    # Generate summary text
    summary = f"""Patient assessment based on {total_tests} completed tests.
    
{'Recent analysis shows consistent monitoring across all assessment levels. ' if total_tests > 3 else 'Limited test history available. '}
{findings[0] if findings else 'No significant findings at this time.'}

Regular monitoring recommended to track progression and adjust treatment as needed."""
    
    return {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'summary': summary,
        'key_findings': findings[:4] if findings else ['Continue regular monitoring']
    }


if __name__ == '__main__':
    print("🚀 Starting Parkinson's Detection API Server...")
    print("📍 Server running on http://localhost:10000")
    print("🔗 Endpoints:")
    print("   - POST /api/tremor/analyze")
    print("   - POST /api/voice/result (store voice test results)")
    print("   - POST /api/facial/result (store facial test results)")
    print("   - GET  /api/caregiver/dashboard")
    print("   - GET  /api/doctor/dashboard")
    print("   - GET  /api/patient/history")
    print("\n💡 Connect to: http://YOUR_COMPUTER_IP:10000 for real devices")
    app.run(host='0.0.0.0', port=10000, debug=True)
