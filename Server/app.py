from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Allow CORS requests from your frontend
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# # Load model and scaler from the parent directory of this file
# base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# model_path = os.path.join(base_dir, 'fnn.pkl')
# scaler_path = os.path.join(base_dir, 'scaler.pkl')

# # Debug print (optional)
# print("Model path:", model_path)
# print("Scaler path:", scaler_path)

# Load them

model = joblib.load("fnn.pkl")
scaler = joblib.load("scaler.pkl")

# Columns used for training
columns = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        print("Input JSON:", input_data)

        input_df = pd.DataFrame([input_data])
        input_df = input_df[columns]

        input_scaled = scaler.transform(input_df)
        predictions = model.predict(input_scaled)

        print("Prediction:", predictions)

        return jsonify({'prediction': int(predictions[0])})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 400
