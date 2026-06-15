from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from disease_info import DISEASE_INFO
import joblib
import numpy as np

# Create app
app = FastAPI()

# Load model files
model = joblib.load("rf_model.pkl")
symptom_index = joblib.load("symptom_index.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Input format
class SymptomsInput(BaseModel):
    symptoms: list[str]

# Home route
@app.get("/")
def home():
    return {
        "message": "Medical AI API is running"
    }
@app.get("/symptoms")
def get_symptoms():
    return sorted(list(symptom_index.keys()))

@app.get("/diseases")
def get_diseases():
    return list(label_encoder.classes_)

# Prediction route
@app.post("/predict")
def predict(data: SymptomsInput):

    # Check if list is empty
    if not data.symptoms:
        raise HTTPException(
            status_code=400,
            detail="Please provide at least one symptom."
        )

    input_vector = np.zeros(len(symptom_index))

    valid_symptoms = []
    invalid_symptoms = []

    for symptom in data.symptoms:

        symptom = symptom.lower().strip()

        if symptom in symptom_index:
           input_vector[symptom_index[symptom]] = 1
           valid_symptoms.append(symptom)
        else:
         invalid_symptoms.append(symptom)

    # Check if no valid symptoms were found

    if len(valid_symptoms) == 0:
        raise HTTPException(
           status_code=400,
           detail={
              "message": "No valid symptoms were provided.",
              "invalid_symptoms": invalid_symptoms
            }
        )

    probabilities = model.predict_proba([input_vector])[0]

    top_indices = np.argsort(probabilities)[-3:][::-1]

    predictions = []

    for idx in top_indices:

        disease = label_encoder.inverse_transform([idx])[0]

        confidence = round(float(probabilities[idx]) * 100, 2)

        info = DISEASE_INFO.get(
            disease.lower(),
            {
                "description": "Information not available.",
                "causes": [],
                "precautions": ["Consult a healthcare professional."],
                "doctor": "General Physician",
                "severity": "Unknown"
            }
        )

        predictions.append({
            "disease": disease,
            "confidence": confidence,
            "description": info["description"],
            "causes": info["causes"],
            "precautions": info["precautions"],
            "doctor": info["doctor"],
            "severity": info["severity"]
        })

    return {
        
    "valid_symptoms": valid_symptoms,
    "invalid_symptoms": invalid_symptoms,
    "predictions": predictions
}
    