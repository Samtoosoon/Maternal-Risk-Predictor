"""
Flask Backend for Maternal Health Risk Prediction
Provides REST API endpoints to predict pregnancy risk using trained ML models.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict import predict_single, load_all_models, EXPECTED_FEATURES

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load models at startup for faster predictions
print("Loading ML models...")
models_map, meta_model = load_all_models()
print(f"Loaded {len(models_map)} model(s). Meta model: {'Yes' if meta_model else 'No'}")


def get_risk_factors(patient_data: dict, risk_score: int) -> list:
    """
    Identify key risk factors contributing to the prediction.
    Returns a list of contributing factors for explainability.
    """
    factors = []
    
    # Age-based risk factors
    age = patient_data.get('age', 0)
    if age < 18:
        factors.append({"factor": "Teenage pregnancy", "severity": "high", "description": f"Mother's age ({age}) is below 18 years"})
    elif age > 35:
        factors.append({"factor": "Advanced maternal age", "severity": "medium", "description": f"Mother's age ({age}) is above 35 years"})
    
    # Blood pressure risk factors
    systolic_bp = patient_data.get('systolic_blood_pressure', 0)
    diastolic_bp = patient_data.get('diastolic_blood_pressure', 0)
    if systolic_bp >= 140 or diastolic_bp >= 90:
        factors.append({"factor": "High blood pressure", "severity": "high", "description": f"BP: {systolic_bp}/{diastolic_bp} mmHg indicates hypertension"})
    elif systolic_bp >= 120 or diastolic_bp >= 80:
        factors.append({"factor": "Elevated blood pressure", "severity": "medium", "description": f"BP: {systolic_bp}/{diastolic_bp} mmHg is elevated"})
    
    # Blood sugar risk factors
    blood_sugar = patient_data.get('blood_sugar', 0)
    if blood_sugar > 140:
        factors.append({"factor": "High blood sugar", "severity": "high", "description": f"Blood glucose level ({blood_sugar} mg/dL) indicates gestational diabetes risk"})
    elif blood_sugar > 100:
        factors.append({"factor": "Elevated blood sugar", "severity": "medium", "description": f"Blood glucose level ({blood_sugar} mg/dL) is elevated"})
    
    # Hemoglobin (anemia) risk factors
    hemoglobin = patient_data.get('hemoglobin', 12)
    if hemoglobin < 7:
        factors.append({"factor": "Severe anemia", "severity": "high", "description": f"Hemoglobin ({hemoglobin} g/dL) indicates severe anemia"})
    elif hemoglobin < 10:
        factors.append({"factor": "Moderate anemia", "severity": "medium", "description": f"Hemoglobin ({hemoglobin} g/dL) indicates moderate anemia"})
    elif hemoglobin < 11:
        factors.append({"factor": "Mild anemia", "severity": "low", "description": f"Hemoglobin ({hemoglobin} g/dL) indicates mild anemia"})
    
    # Heart rate risk factors
    heart_rate = patient_data.get('heart_rate', 80)
    if heart_rate > 100:
        factors.append({"factor": "Elevated heart rate", "severity": "medium", "description": f"Heart rate ({heart_rate} bpm) is above normal range"})
    elif heart_rate < 60:
        factors.append({"factor": "Low heart rate", "severity": "low", "description": f"Heart rate ({heart_rate} bpm) is below normal range"})
    
    # BMI risk factors
    bmi = patient_data.get('bmi', 22)
    if bmi > 30:
        factors.append({"factor": "Obesity", "severity": "medium", "description": f"BMI ({bmi:.1f}) indicates obesity"})
    elif bmi < 18.5:
        factors.append({"factor": "Underweight", "severity": "medium", "description": f"BMI ({bmi:.1f}) indicates underweight"})
    
    # ANC visits risk factors
    anc_visits = patient_data.get('anc_visits', 4)
    if anc_visits < 4:
        factors.append({"factor": "Insufficient prenatal care", "severity": "medium", "description": f"Only {anc_visits} ANC visits (WHO recommends 4+)"})
    
    # Location risk factors
    location = patient_data.get('location', 1)
    if location == 0:
        factors.append({"factor": "Rural location", "severity": "low", "description": "Limited access to healthcare facilities"})
    
    # Parity risk factors
    parity = patient_data.get('parity', 0)
    if parity > 4:
        factors.append({"factor": "Grand multiparity", "severity": "medium", "description": f"High number of previous pregnancies ({parity})"})
    elif parity == 0:
        factors.append({"factor": "First pregnancy", "severity": "low", "description": "Primigravida - first-time mother"})
    
    # Sort by severity
    severity_order = {'high': 0, 'medium': 1, 'low': 2}
    factors.sort(key=lambda x: severity_order.get(x['severity'], 3))
    
    return factors


def get_recommendations(risk_category: str, factors: list) -> list:
    """
    Generate recommendations based on risk category and contributing factors.
    Guidelines sourced from WHO, ACOG, and NICE clinical standards.
    """
    recommendations = []
    
    if risk_category == 'High':
        recommendations.append({
            "priority": "urgent",
            "action": "Immediate referral to high-risk obstetric care",
            "description": "This pregnancy requires specialized care and close monitoring",
            "source": "WHO Maternal Health Guidelines"
        })
        recommendations.append({
            "priority": "urgent",
            "action": "Schedule frequent check-ups",
            "description": "Weekly or bi-weekly visits recommended for high-risk pregnancies",
            "source": "ACOG Practice Bulletin"
        })
    elif risk_category == 'Medium':
        recommendations.append({
            "priority": "important",
            "action": "Enhanced monitoring required",
            "description": "More frequent prenatal visits and additional tests recommended",
            "source": "NICE Antenatal Care Guidelines"
        })
    
    # Factor-specific recommendations
    for factor in factors:
        if 'anemia' in factor['factor'].lower():
            recommendations.append({
                "priority": "high" if factor['severity'] == 'high' else "medium",
                "action": "Iron and folic acid supplementation",
                "description": "Daily iron supplementation (30-60mg) and folic acid recommended",
                "source": "WHO Antenatal Care Recommendations"
            })
        if 'blood pressure' in factor['factor'].lower() and factor['severity'] == 'high':
            recommendations.append({
                "priority": "high",
                "action": "Monitor for pre-eclampsia symptoms",
                "description": "Watch for headaches, visual changes, epigastric pain, and swelling",
                "source": "ACOG Hypertension in Pregnancy Guidelines"
            })
        if 'blood sugar' in factor['factor'].lower():
            recommendations.append({
                "priority": "high",
                "action": "Glucose tolerance test and dietary management",
                "description": "Screen with 75g OGTT and provide dietary counseling",
                "source": "IADPSG Gestational Diabetes Guidelines"
            })
        if 'age' in factor['factor'].lower() and 'advanced' in factor['factor'].lower():
            recommendations.append({
                "priority": "medium",
                "action": "Consider genetic screening and fetal monitoring",
                "description": "Offer cell-free DNA screening and detailed ultrasound",
                "source": "ACOG Advanced Maternal Age Guidelines"
            })
        if 'obesity' in factor['factor'].lower():
            recommendations.append({
                "priority": "medium",
                "action": "Nutritional counseling and weight management",
                "description": "Limit gestational weight gain per IOM guidelines",
                "source": "IOM Weight Gain Guidelines"
            })
    
    return recommendations[:5]  # Limit to top 5 recommendations


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": len(models_map) > 0,
        "meta_model_available": meta_model is not None
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict maternal health risk from patient data.
    
    Expected JSON body:
    {
        "age": 25,
        "systolic_blood_pressure": 120,
        "diastolic_blood_pressure": 80,
        "blood_sugar": 85,
        "hemoglobin": 12,
        "heart_rate": 75,
        "bmi": 22.5,
        "parity": 1,
        "anc_visits": 4,
        "location": 1  # 0: Rural, 1: Urban
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate required fields
        required_fields = ['age', 'systolic_blood_pressure', 'diastolic_blood_pressure', 
                          'blood_sugar', 'hemoglobin', 'heart_rate']
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Set defaults for optional fields
        patient_data = {
            'age': float(data.get('age', 25)),
            'systolic_blood_pressure': float(data.get('systolic_blood_pressure', 120)),
            'diastolic_blood_pressure': float(data.get('diastolic_blood_pressure', 80)),
            'blood_sugar': float(data.get('blood_sugar', 85)),
            'hemoglobin': float(data.get('hemoglobin', 12)),
            'heart_rate': float(data.get('heart_rate', 75)),
            'bmi': float(data.get('bmi', 22.5)),
            'parity': int(data.get('parity', 0)),
            'anc_visits': int(data.get('anc_visits', 4)),
            'location': int(data.get('location', 1))
        }
        
        # Convert units for model compatibility
        # UCI dataset uses blood_sugar in mmol/L (range 6-19), UI uses mg/dL (range 70-300)
        # Conversion: mmol/L = mg/dL / 18
        model_data = patient_data.copy()
        model_data['blood_sugar'] = patient_data['blood_sugar'] / 18.0
        
        # UCI dataset has location as text ('rural', 'urban'), model expects encoded values
        # Keep as numeric since preprocessor will handle it
        model_data['location'] = 'rural' if patient_data['location'] == 0 else 'urban'
        
        # Get prediction from ML model
        result = predict_single(model_data)
        
        # Get risk factors and recommendations
        risk_factors = get_risk_factors(patient_data, result['score'])
        recommendations = get_recommendations(result['category'], risk_factors)
        
        response = {
            "risk_score": result['score'],
            "risk_category": result['category'],
            "risk_factors": risk_factors,
            "recommendations": recommendations,
            "input_data": patient_data
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/sample-profiles', methods=['GET'])
def get_sample_profiles():
    """
    Return sample patient profiles for demonstration.
    """
    profiles = [
        {
            "name": "Teenage Mother with Anemia",
            "description": "17-year-old with low hemoglobin and limited prenatal care",
            "data": {
                "age": 17,
                "systolic_blood_pressure": 125,
                "diastolic_blood_pressure": 82,
                "blood_sugar": 95,
                "hemoglobin": 8.5,
                "heart_rate": 88,
                "bmi": 19.5,
                "parity": 0,
                "anc_visits": 2,
                "location": 0
            }
        },
        {
            "name": "Healthy Adult Mother",
            "description": "28-year-old with normal parameters and good prenatal care",
            "data": {
                "age": 28,
                "systolic_blood_pressure": 118,
                "diastolic_blood_pressure": 76,
                "blood_sugar": 82,
                "hemoglobin": 12.5,
                "heart_rate": 72,
                "bmi": 23.0,
                "parity": 1,
                "anc_visits": 6,
                "location": 1
            }
        },
        {
            "name": "High-Risk Older Mother",
            "description": "38-year-old with hypertension and elevated blood sugar",
            "data": {
                "age": 38,
                "systolic_blood_pressure": 148,
                "diastolic_blood_pressure": 95,
                "blood_sugar": 155,
                "hemoglobin": 10.5,
                "heart_rate": 92,
                "bmi": 31.2,
                "parity": 3,
                "anc_visits": 4,
                "location": 1
            }
        },
        {
            "name": "Rural Mother with Multiple Risk Factors",
            "description": "32-year-old from rural area with multiple moderate risks",
            "data": {
                "age": 32,
                "systolic_blood_pressure": 135,
                "diastolic_blood_pressure": 88,
                "blood_sugar": 110,
                "hemoglobin": 9.8,
                "heart_rate": 85,
                "bmi": 26.5,
                "parity": 4,
                "anc_visits": 3,
                "location": 0
            }
        }
    ]
    
    return jsonify(profiles)


@app.route('/api/feature-info', methods=['GET'])
def get_feature_info():
    """
    Return information about input features for the UI.
    """
    features = {
        "age": {
            "label": "Age (years)",
            "description": "Mother's age in years",
            "min": 13,
            "max": 50,
            "default": 25,
            "type": "number",
            "normal_range": "18-35 years"
        },
        "systolic_blood_pressure": {
            "label": "Systolic BP (mmHg)",
            "description": "Upper blood pressure reading",
            "min": 70,
            "max": 200,
            "default": 120,
            "type": "number",
            "normal_range": "90-120 mmHg"
        },
        "diastolic_blood_pressure": {
            "label": "Diastolic BP (mmHg)",
            "description": "Lower blood pressure reading",
            "min": 40,
            "max": 130,
            "default": 80,
            "type": "number",
            "normal_range": "60-80 mmHg"
        },
        "blood_sugar": {
            "label": "Blood Sugar (mg/dL)",
            "description": "Fasting blood glucose level",
            "min": 50,
            "max": 300,
            "default": 85,
            "type": "number",
            "normal_range": "70-100 mg/dL"
        },
        "hemoglobin": {
            "label": "Hemoglobin (g/dL)",
            "description": "Hemoglobin level in blood",
            "min": 5,
            "max": 17,
            "default": 12,
            "type": "number",
            "normal_range": "11-14 g/dL (pregnancy)"
        },
        "heart_rate": {
            "label": "Heart Rate (bpm)",
            "description": "Resting heart rate",
            "min": 40,
            "max": 150,
            "default": 75,
            "type": "number",
            "normal_range": "60-100 bpm"
        },
        "bmi": {
            "label": "BMI",
            "description": "Body Mass Index",
            "min": 15,
            "max": 50,
            "default": 22.5,
            "type": "number",
            "normal_range": "18.5-24.9"
        },
        "parity": {
            "label": "Previous Pregnancies",
            "description": "Number of previous pregnancies",
            "min": 0,
            "max": 10,
            "default": 0,
            "type": "number",
            "normal_range": "0-4"
        },
        "anc_visits": {
            "label": "ANC Visits",
            "description": "Number of antenatal care visits",
            "min": 0,
            "max": 15,
            "default": 4,
            "type": "number",
            "normal_range": "4+ recommended"
        },
        "location": {
            "label": "Location",
            "description": "Rural (0) or Urban (1)",
            "options": [
                {"value": 0, "label": "Rural"},
                {"value": 1, "label": "Urban"}
            ],
            "default": 1,
            "type": "select"
        }
    }
    
    return jsonify(features)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
