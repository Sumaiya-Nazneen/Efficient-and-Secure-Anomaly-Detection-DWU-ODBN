from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime
import pandas as pd
import tensorflow as tf
import os



app = Flask(__name__)
app.secret_key = 'zmdb'  # Change to a random secret key

DATABASE = 'users.db'


BASE_DIR = "models/iot"


def load_device_components(device_name):
    """Load model, scaler, and label encoder for the selected device"""
    device_path = os.path.join(BASE_DIR, device_name)
    model_path = os.path.join(device_path, "final_network_model.h5")
    print(model_path)
    scaler_path = os.path.join(device_path, "scaler.pkl")
    encoder_path = os.path.join(device_path, "type_encoder.pkl")

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(encoder_path)

    return model, scaler, le




def predict_attack_type(new_data, features, device):
    """Predict the attack type for a specific IoT device"""
    model, scaler, le = load_device_components(device)
    df = pd.DataFrame([new_data])
    scaled = scaler.transform(df[features])
    pred = model.predict(scaled)
    pred_class = np.argmax(pred, axis=1)[0]
    return le.inverse_transform([pred_class])[0]


# Register custom layer if using DualWeightDense
class DualWeightDense(tf.keras.layers.Layer):
    def __init__(self, units, dual_factor=0.5, **kwargs):
        super(DualWeightDense, self).__init__(**kwargs)
        self.units = units
        self.dual_factor = dual_factor

    def build(self, input_shape):
        self.w1 = self.add_weight(shape=(input_shape[-1], self.units),
                                  initializer='random_normal',
                                  trainable=True)
        self.w2 = self.add_weight(shape=(input_shape[-1], self.units),
                                  initializer='random_normal',
                                  trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        output = tf.matmul(inputs, self.dual_factor * self.w1 + (1 - self.dual_factor) * self.w2) + self.b
        return tf.nn.relu(output)
    



feature_names = [
    'src_port', 'dst_port', 'proto', 'service', 'duration', 'src_bytes',
       'dst_bytes', 'conn_state', 'missed_bytes', 'src_pkts', 'src_ip_bytes',
       'dst_pkts', 'dst_ip_bytes', 'dns_query', 'dns_qclass', 'dns_qtype',
       'dns_rcode', 'dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected',
       'ssl_version', 'ssl_cipher', 'ssl_resumed', 'ssl_established',
       'ssl_subject', 'ssl_issuer', 'http_trans_depth', 'http_method',
       'http_version', 'http_request_body_len', 'http_response_body_len',
       'http_status_code', 'http_user_agent', 'http_orig_mime_types',
       'http_resp_mime_types', 'weird_name', 'weird_addl', 'weird_notice'
]


# Load your fitted scaler, LabelEncoder, and model once at startup
scaler = joblib.load('models/network/network_scaler.pkl')
model = load_model('models/network/dwu_obn_network_model.h5', custom_objects={'DualWeightDense': DualWeightDense})

# Path where column-wise LabelEncoders are saved
label_encoders_dir = 'models/network/label_encoders/'


def load_label_encoders():
    encoders = {}
    for filename in os.listdir(label_encoders_dir):
        if filename.endswith('.pkl'):
            col_name = filename.replace('_label_encoder.pkl', '')
            encoders[col_name] = joblib.load(os.path.join(label_encoders_dir, filename))
    return encoders

network_label_encoders = load_label_encoders()

# windows data

# Load model and scaler once at startup
models = {
    'windows_7_model': load_model('models/windows_model/windows_7_model.h5'),
    'windows_10_model': load_model('models/windows_model/windows_10_model.h5')
}
scalers = {
    'windows_7_model': joblib.load('models/windows_model/df_7_scaler.pkl'),
    'windows_10_model': joblib.load('models/windows_model/df_10_scaler.pkl')
}
label_encoders = {
    'windows_7_model': joblib.load('models/windows_model/df_7_encoder.pkl'),
    'windows_10_model': joblib.load('models/windows_model/df_10_encoder.pkl')
}

attack_details = {
    "normal": {
        "description": "Normal system activity with no detected threat.",
        "recommended_actions": [
            "Continue routine monitoring",
            "Maintain system updates",
            "Review security logs periodically"
        ],
        "severity": "Low"
    },
    "ddos": {
        "description": "Distributed Denial-of-Service (DDoS) attacks attempt to overwhelm network resources and cause downtime.",
        "recommended_actions": [
            "Block suspicious IP addresses",
            "Enable rate limiting on incoming traffic",
            "Increase firewall and network monitoring capacity"
        ],
        "severity": "High"
    },
    "password": {
        "description": "Password-based attacks involve guessing, brute force, or dictionary attacks to gain unauthorized access.",
        "recommended_actions": [
            "Enforce strong password policies",
            "Enable multi-factor authentication (MFA)",
            "Monitor and block multiple failed login attempts"
        ],
        "severity": "Medium"
    },
    "xss": {
        "description": "Cross-Site Scripting (XSS) allows attackers to inject malicious scripts into web pages viewed by other users.",
        "recommended_actions": [
            "Sanitize and validate all user inputs",
            "Use Content Security Policy (CSP)",
            "Keep web frameworks and libraries up-to-date"
        ],
        "severity": "Medium"
    },
    "injection": {
        "description": "Injection attacks (SQL, command, or code injection) exploit vulnerabilities in input handling to execute malicious code.",
        "recommended_actions": [
            "Use parameterized queries or prepared statements",
            "Sanitize user input and escape special characters",
            "Patch vulnerable software immediately"
        ],
        "severity": "High"
    },
    "dos": {
        "description": "Denial-of-Service (DoS) attacks disrupt network or service availability by overwhelming resources.",
        "recommended_actions": [
            "Implement rate limiting",
            "Monitor and filter traffic at the network perimeter",
            "Deploy intrusion detection systems (IDS)"
        ],
        "severity": "High"
    },
    "scanning": {
        "description": "Scanning attacks involve probing systems to identify open ports or exploitable vulnerabilities.",
        "recommended_actions": [
            "Block repeated scanning IPs",
            "Use network intrusion detection systems",
            "Conduct regular vulnerability assessments"
        ],
        "severity": "Medium"
    },
    "mitm": {
        "description": "Man-in-the-Middle (MITM) attacks intercept communications between two systems to steal or alter information.",
        "recommended_actions": [
            "Use HTTPS/TLS encryption everywhere",
            "Avoid public or unsecured Wi-Fi for sensitive data",
            "Implement strong authentication and session management"
        ],
        "severity": "High"
    },
    "backdoor": {
        "description": "Backdoor attacks involve unauthorized remote access channels installed by attackers or malicious software.",
        "recommended_actions": [
            "Perform full malware scans",
            "Audit network connections and processes",
            "Reinstall compromised systems from trusted backups"
        ],
        "severity": "Critical"
    },
    "ransomware": {
        "description": "Ransomware encrypts user files or systems and demands payment for decryption.",
        "recommended_actions": [
            "Disconnect infected systems from the network immediately",
            "Do not pay the ransom — report to authorities",
            "Restore from secure backups and update all software"
        ],
        "severity": "Critical"
    },
    "other_attack": {
        "description": "Other or unknown types of security threats, including hybrid or emerging attack patterns.",
        "recommended_actions": [
            "Review recent system and network logs",
            "Update antivirus and firewall configurations",
            "Consult cybersecurity experts for deeper analysis"
        ],
        "severity": "Variable - assess case-by-case"
    }
}



def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
    print('Database Initialized.')

df_10 = pd.read_csv('data/Train_Test_Windows_dataset/Train_Test_Windows_10.csv')
df_7 = pd.read_csv(
    'data/Train_Test_Windows_dataset/Train_Test_Windows_7.csv',
    on_bad_lines='skip',
    engine='python'
)


def predict_anomaly(data, os_type,
                    model_path_base='models/windows_model/windows',
                    scaler_path_base='models/windows_model/df',
                    encoder_path_base='models/windows_model/df'):
    """
    Predicts the anomaly type for a single data point.
    """

    # --- Step 1: Convert input to DataFrame ---
    if isinstance(data, list):
        # If you have predefined training DataFrames, use them for column order
        if os_type == '10':
            feature_columns = df_10.drop(columns=['label', 'type']).columns.tolist()
        elif os_type == '7':
            feature_columns = df_7.drop(columns=['label', 'type']).columns.tolist()
        else:
            raise ValueError("os_type must be '10' or '7'")

        if len(data) != len(feature_columns):
            raise ValueError(f"Input length ({len(data)}) does not match features ({len(feature_columns)}).")

        data = pd.DataFrame([data], columns=feature_columns)

    elif isinstance(data, dict):
        data = pd.DataFrame([data])
    elif not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a list, dict, or DataFrame.")

    # --- Step 2: Load model, scaler, and label encoder ---
    model = load_model(f'{model_path_base}_{os_type}_model.h5')
    scaler = joblib.load(f'{scaler_path_base}_{os_type}_scaler.pkl')
    label_encoder = joblib.load(f'{encoder_path_base}_{os_type}_encoder.pkl')

    # --- Step 3: Clean input data ---
    for col in data.columns:
        if data[col].dtype == 'object':  # only convert object columns
            data[col] = pd.to_numeric(data[col], errors='coerce').astype(float)
    data.fillna(0.0, inplace=True)

    # --- Step 4: Scale the features ---
    data_scaled = scaler.transform(data)

    # --- Step 5: Make prediction ---
    prediction_prob = model.predict(data_scaled)
    predicted_class_index = np.argmax(prediction_prob, axis=1)[0]
    confidence = float(prediction_prob[0][predicted_class_index])

    # --- Step 6: Decode prediction ---
    predicted_anomaly_type = label_encoder.inverse_transform([predicted_class_index])[0]

    return predicted_anomaly_type, confidence





@app.route('/iot')
def iot():
    return render_template('iot.html')


@app.route('/predict-iot', methods=['POST'])
def predict_iot():
    try:
        # Parse JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        device = data.get('device')
        if not device:
            return jsonify({"error": "Device not specified"}), 400

        # Collect device-specific input features
        if device == 'fridge':
            inputs = {
                'fridge_temperature': float(data['fridge_temperature']),
                'temp_condition': 1 if data['temp_condition'] == 'high' else 0,
                'hour': int(data['hour']),
                'day': int(data['day']),
                'month': int(data['month']),
                'weekday': int(data['weekday'])
            }
            features = ['fridge_temperature', 'temp_condition', 'hour', 'day', 'month', 'weekday']

        elif device == 'door':
            inputs = {
                'door_state': int(data['door_state']),
                'sphone_signal': int(data['sphone_signal']),
                'hour': int(data['hour']),
                'day': int(data['day']),
                'month': int(data['month']),
                'weekday': int(data['weekday'])
            }
            features = ['door_state', 'sphone_signal', 'hour', 'day', 'month', 'weekday']

        elif device == 'thermostat':
            inputs = {
                'current_temperature': float(data['current_temperature']),
                'thermostat_status': int(data['thermostat_status']),
                'hour': int(data['hour']),
                'day': int(data['day']),
                'month': int(data['month']),
                'weekday': int(data['weekday'])
            }
            features = ['current_temperature', 'thermostat_status', 'hour', 'day', 'month', 'weekday']

        elif device == 'motion':
            inputs = {
                'motion_status': int(data['motion_status']),
                'light_status': int(data['light_status']),
                'hour': int(data['hour']),
                'day': int(data['day']),
                'month': int(data['month']),
                'weekday': int(data['weekday'])
            }
            features = ['motion_status', 'light_status', 'hour', 'day', 'month', 'weekday']

        elif device == 'weather':
            inputs = {
                'temperature': float(data['temperature']),
                'pressure': float(data['pressure']),
                'humidity': float(data['humidity']),
                'hour': int(data['hour']),
                'day': int(data['day']),
                'month': int(data['month']),
                'weekday': int(data['weekday'])
            }
            features = ['temperature', 'pressure', 'humidity', 'hour', 'day', 'month', 'weekday']

        elif device == 'gps':
            inputs = {
                'latitude': float(data['latitude']),
                'longitude': float(data['longitude']),
                'hour': int(data['hour']),
                'day': int(data['day']),
                'month': int(data['month']),
                'weekday': int(data['weekday'])
            }
            features = ['latitude', 'longitude', 'hour', 'day', 'month', 'weekday']

        else:
            return jsonify({"error": "Invalid device selected"}), 400

        # Run prediction using your helper function
        model, scaler, le = load_device_components(device)
        df = pd.DataFrame([inputs])
        scaled = scaler.transform(df[features])
        prediction = model.predict(scaled)
        pred_class_index = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
        predicted_attack = le.inverse_transform([pred_class_index])[0]

        # Map to attack details (if available)
        details = attack_details.get(predicted_attack, {
            "description": "No details available for this class.",
            "recommended_actions": ["Monitor the system for further anomalies."],
            "severity": "Unknown"
        })

        # Return JSON response
        return jsonify({
            "device": device,
            "input_data": inputs,
            "predicted_attack": predicted_attack,
            "confidence": round(confidence, 4),
            "description": details["description"],
            "recommended_actions": details["recommended_actions"],
            "severity": details["severity"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/')
def index():
    return render_template('index.html', user=session.get('user_email'))

@app.route('/network')
def network():
    return render_template('network.html')
@app.route('/predict_network', methods=['POST'])
def predict_network():
    try:
        data = request.get_json()
        features = data.get('features')
        if not features or not isinstance(features, list):
            return jsonify({'error': 'Features must be a non-empty list'}), 400
        print(len(feature_names),len(features))
        print(features)

        # Convert list to DataFrame with known feature column order
        input_df = pd.DataFrame([features], columns=feature_names)
        print(input_df)

        # Encode categorical columns using saved LabelEncoders
        for col in input_df.select_dtypes(include='object').columns:
            if col in network_label_encoders:
                le = network_label_encoders[col]
                input_df[col] = le.transform(input_df[col].astype(str))
            else:
                return jsonify({'error': f'Encoder for column "{col}" not found'}), 400

        # Scale the features
        input_scaled = scaler.transform(input_df)

        # Predict anomaly class (binary classification)
        prediction_proba = model.predict(input_scaled)
        prediction = (prediction_proba > 0.5).astype("int32")[0][0]

        label = "Anomaly Detected" if prediction == 1 else "Normal"
        confidence = float(prediction_proba[0][0])
        if label == "Normal":
            recommendation = [
        "No suspicious activity detected.",
        "Continue regular network monitoring.",
        "Ensure firewall and antivirus remain up-to-date."
    ]
        else:
            recommendation = [
        "Anomaly detected — possible malicious or abnormal network behavior.",
        "Isolate the affected system or IP for investigation.",
        "Review recent traffic logs for unusual spikes or patterns.",
        "Scan for malware or unauthorized access.",
        "Block suspicious IP addresses and monitor further activity."
    ]


        response = {
            'prediction': label,
            'confidence': confidence,
              'recommendations': recommendation
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    input_features = data.get('features')
    os_type = data.get('os_type')  # '10' or '7'
    print(os_type)

    if not input_features:
        return jsonify({'error': 'Missing features array'}), 400
    if os_type not in ['7', '10']:
        return jsonify({'error': 'Invalid or missing os_type (must be "7" or "10")'}), 400

    try:
        predicted_label, confidence = predict_anomaly(input_features, os_type)

        details = attack_details.get(predicted_label, {
            "description": "No details available for this class.",
            "recommended_actions": [],
            "severity": "Unknown"
        })

        response = {
            'prediction': predicted_label,
            'confidence': round(confidence, 4),
            'description': details['description'],
            'recommended_actions': details['recommended_actions'],
            'severity': details['severity'],
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'os_type': os_type
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/windows")
def windows():
    return render_template('windows.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return jsonify({'message': 'Email and password are required.'}), 400

    hashed_pw = generate_password_hash(password)
    try:
        with get_db() as db:
            db.execute('INSERT INTO users (email, password) VALUES (?, ?)', (email, hashed_pw))
        return jsonify({'message': 'Registration successful. Please log in.'}), 200
    except sqlite3.IntegrityError:
        return jsonify({'message': 'Email already registered.'}), 400

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return jsonify({'message': 'Email and password are required.'}), 400

    with get_db() as db:
        user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()

    if user and check_password_hash(user['password'], password):
        session['user_email'] = user['email']
        return jsonify({'message': 'Logged in successfully.'}), 200
    else:
        return jsonify({'message': 'Invalid email or password.'}), 401

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')

@app.route('/logout', methods=['GET'])
def logout():
    session.pop('user_email', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
