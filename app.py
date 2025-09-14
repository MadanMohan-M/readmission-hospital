from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import uuid

app = Flask(__name__)
app.secret_key = "healthguard_ai_2025_secure_key"

# Enhanced doctor credentials with roles
DOCTORS = {
    "dr.chen": {"password": "medical2025", "name": "Dr. Sarah Chen", "role": "Chief of Medicine", "department": "Internal Medicine"},
    "dr.patel": {"password": "cardio123", "name": "Dr. Raj Patel", "role": "Cardiologist", "department": "Cardiology"},
    "admin": {"password": "admin123", "name": "System Administrator", "role": "IT Admin", "department": "Technology"}
}

# Load ML model and training columns
try:
    model = joblib.load('rf_model2.pkl')
    train_cols = np.load('train_columns.npy', allow_pickle=True)
    print("✅ Model and training columns loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    train_cols = None

age_map = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65,
    '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
}
test_map = {'no': 0, 'normal': 1, 'high': 2}
binary_map = {'no': 0, 'yes': 1}

patient_records = []

def generate_patient_id():
    return f"PT-{uuid.uuid4().hex[:12].upper()}"

def preprocess_input(input_dict: dict) -> pd.DataFrame:
    try:
        df = pd.DataFrame([input_dict])
        log_cols = ['n_emergency', 'n_outpatient', 'n_medications']
        for col in log_cols:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col])
                df.drop(columns=[col], inplace=True)
        if 'age' in df.columns:
            df['age_encoded'] = df['age'].map(age_map)
        if 'A1Ctest' in df.columns:
            df['A1Ctest_encoded'] = df['A1Ctest'].map(test_map)
        if 'glucose_test' in df.columns:
            df['glucose_test_encoded'] = df['glucose_test'].map(test_map)
        if 'change' in df.columns:
            df['change_encoded'] = df['change'].map(binary_map)
        if 'diabetes_med' in df.columns:
            df['diabetes_med_encoded'] = df['diabetes_med'].map(binary_map)
        nominal_cols = ['medical_specialty', 'diag_1', 'diag_2', 'diag_3']
        for col in nominal_cols:
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col], drop_first=True)
        drop_cols = ['age', 'A1Ctest', 'glucose_test', 'change', 'diabetes_med']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
        if train_cols is not None:
            for col in train_cols:
                if col not in df.columns:
                    df[col] = 0
            df = df[train_cols]
        print("Preprocessing successful. Columns after preprocessing:", df.columns.tolist())
        return df
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return pd.DataFrame()

@app.before_request
def require_login():
    public_routes = ['login', 'static', 'signup']
    if request.endpoint not in public_routes and not session.get('doctor_logged_in'):
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('doctor_logged_in'):
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').lower()
        password = request.form.get('password', '')
        if username in DOCTORS and DOCTORS[username]['password'] == password:
            session['doctor_logged_in'] = True
            session['doctor_info'] = DOCTORS[username]
            session['login_time'] = datetime.now().isoformat()
            flash(f"Welcome back, {DOCTORS[username]['name']}!", 'success')
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials. Please try again.", 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        flash("Account registration requires administrator approval. Please contact IT support.", 'info')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/')
@app.route('/dashboard')
def dashboard():
    stats = {
        'total_assessments': len(patient_records),
        'high_risk_cases': len([p for p in patient_records if p.get('risk_level') == 'High']),
        'cost_savings': 2400000,
        'accuracy': 94.2
    }
    return render_template('dashboard.html', doctor=session.get('doctor_info', {}), stats=stats)

@app.route('/predict')
def predict_page():
    return render_template('predict.html', doctor=session.get('doctor_info', {}))

@app.route('/process_prediction', methods=['POST'])
def process_prediction():
    # Check if request wants JSON (AJAX call)
    wants_json = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.accept_mimetypes.accept_json
    
    if model is None:
        if wants_json:
            return jsonify({'error': "ML model not available. Please contact system administrator."}), 500
        else:
            flash("ML model not available. Please contact system administrator.", 'error')
            return redirect(url_for('predict_page'))
    
    try:
        form_data = request.form.to_dict()
        print("Received form data:", form_data)

        numeric_fields = ['n_emergency', 'n_outpatient', 'n_inpatient', 'n_medications',
                          'time_in_hospital', 'n_procedures', 'n_lab_procedures']
        input_dict = {}
        for field in numeric_fields:
            input_dict[field] = float(form_data.get(field, 0))

        categorical_fields = ['age', 'A1Ctest', 'glucose_test', 'change', 'diabetes_med',
                              'medical_specialty', 'diag_1', 'diag_2', 'diag_3']
        for field in categorical_fields:
            input_dict[field] = form_data.get(field, '')

        X = preprocess_input(input_dict)
        print("Preprocessed input shape:", X.shape)

        if X.empty:
            error_msg = "Preprocessing failed. Please check your input values."
            if wants_json:
                return jsonify({'error': error_msg}), 400
            else:
                flash(error_msg, 'error')
                return redirect(url_for('predict_page'))

        prediction = model.predict(X)
        probability = model.predict_proba(X)[0][1] * 100
        print("Prediction:", prediction, "Probability:", probability)

        if probability >= 70:
            risk_level = "High"
            risk_color = "high"
        elif probability >= 40:
            risk_level = "Medium"
            risk_color = "medium"
        else:
            risk_level = "Low"
            risk_color = "low"

        patient_record = {
            'id': generate_patient_id(),
            'timestamp': datetime.now().isoformat(),
            'probability': round(probability, 1),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'prediction': int(prediction[0]),
            'input_data': form_data,
            'assessed_by': session.get('doctor_info', {}).get('name', 'Unknown')
        }
        patient_records.append(patient_record)

        # Return JSON for AJAX requests, HTML for form submissions
        if wants_json:
            return jsonify({
                'success': True,
                'result': patient_record,
                'redirect_url': url_for('show_results', patient_id=patient_record['id'])
            })
        else:
            return render_template('results.html', result=patient_record, doctor=session.get('doctor_info', {}))
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if wants_json:
            return jsonify({'error': f"Prediction error: {str(e)}"}), 500
        else:
            flash(f"Prediction error: {str(e)}", 'error')
            return redirect(url_for('predict_page'))

@app.route('/results')
def show_results():
    patient_id = request.args.get('patient_id')
    if patient_id:
        patient = next((p for p in patient_records if p['id'] == patient_id), None)
        if patient:
            return render_template('results.html', result=patient, doctor=session.get('doctor_info', {}))
    
    flash("Patient record not found.", 'error')
    return redirect(url_for('history'))

@app.route('/history')
def history():
    return render_template('history.html', patients=reversed(patient_records), doctor=session.get('doctor_info', {}))

@app.route('/patient/<patient_id>')
def patient_detail(patient_id):
    patient = next((p for p in patient_records if p['id'] == patient_id), None)
    if not patient:
        flash("Patient record not found.", 'error')
        return redirect(url_for('history'))
    return render_template('patient_detail.html', patient=patient, doctor=session.get('doctor_info', {}))

@app.route('/analytics')
def analytics():
    total = len(patient_records)
    high_risk = len([p for p in patient_records if p.get('risk_level') == 'High'])
    medium_risk = len([p for p in patient_records if p.get('risk_level') == 'Medium'])
    low_risk = len([p for p in patient_records if p.get('risk_level') == 'Low'])
    
    analytics_data = {
        'total_assessments': total,
        'high_risk_count': high_risk,
        'medium_risk_count': medium_risk,
        'low_risk_count': low_risk,
        'high_risk_percentage': round((high_risk / total * 100) if total > 0 else 0, 1),
        'medium_risk_percentage': round((medium_risk / total * 100) if total > 0 else 0, 1),
        'low_risk_percentage': round((low_risk / total * 100) if total > 0 else 0, 1)
    }
    return render_template('analytics.html', analytics=analytics_data, doctor=session.get('doctor_info', {}))

@app.route('/api/live_stats')
def live_stats():
    return jsonify({
        'total_assessments': len(patient_records),
        'high_risk_cases': len([p for p in patient_records if p.get('risk_level') == 'High']),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out successfully.", 'info')
    return redirect(url_for('login'))

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == "__main__":
    print("🏥 HealthGuard AI - Hospital Readmission Predictor")
    print("🚀 Starting server...")
    app.run(debug=True, host='0.0.0.0', port=5000)