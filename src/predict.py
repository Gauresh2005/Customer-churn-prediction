from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"
THRESHOLD = 0.40

model = joblib.load(MODEL_PATH)

def predict_churn(customer_data):
    missing_features = [
        feature for feature in REQUIRED_FEATURES 
        if feature not in customer_data
    ]

    if missing_features:
        raise ValueError(f"Data doesn't include these features: {missing_features}")

    extra_features = [
        feature for feature in customer_data
        if feature not in REQUIRED_FEATURES
    ]

    if extra_features:
        raise ValueError(f"Unexpected features: {extra_features}")

    customer_df = pd.DataFrame([customer_data])

    churn_probability = float(model.predict_proba(customer_df)[0, 1])

    prediction = int(churn_probability >= THRESHOLD)

    return {
        "churn_probability": churn_probability,
        "prediction": prediction
    }

if __name__ == "__main__":
    customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 350.0
    }

REQUIRED_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]
