from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_churn

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0"
)

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post('/predict')
def predict(data: CustomerData):
    customer_data = data.model_dump()
    result = predict_churn(customer_data)

    return {
        "churn_probability": result["churn_probability"],
        "prediction": result["prediction"],
        "prediction_label": "Churn" if result["prediction"] == 1 else "No Churn"
    }

@app.get('/')
def health_check():
    return {
        "status": "healthy",
        "message": "Customer churn prediction api is running..."
    }