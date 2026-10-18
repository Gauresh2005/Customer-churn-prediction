# Customer Churn Prediction

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to leave the service. The project uses a recall-focused Logistic Regression model and exposes predictions through a FastAPI REST API, packaged with Docker and deployed on Render.

## Live Demo

- **Live API:** `https://customer-churn-prediction-api-e0go.onrender.com/`
- **API Documentation (Swagger):** `https://customer-churn-prediction-api-e0go.onrender.com/docs`
- **GitHub Repository:** https://github.com/Gauresh2005/Customer-churn-prediction

## Project Overview

Customer churn is an important business problem for telecommunications companies. Identifying customers who are likely to leave helps businesses prioritize retention efforts and improve customer satisfaction.

This project develops a machine learning pipeline to estimate churn probability from customer demographics, service subscriptions, contract details, tenure, and billing information.

The model prioritizes **recall** to identify more potential churners, accepting a higher number of false-positive predictions.

## Tech Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy, Matplotlib
- **Machine Learning:** Scikit-learn
- **Model Serialization:** Joblib
- **API:** FastAPI, Pydantic, Uvicorn
- **Containerization:** Docker
- **Deployment:** Render
- **Version Control:** Git and GitHub

## Machine Learning Workflow

1. **Exploratory Data Analysis:** Examined churn distribution and relationships between customer features and churn.
2. **Data Cleaning:** Removed the customer identifier, converted `TotalCharges` to numeric values, and handled missing values.
3. **Preprocessing:** Applied suitable transformations to numerical and categorical features using a machine learning pipeline.
4. **Train-Test Split:** Used a stratified split to preserve the class distribution.
5. **Model Training:** Compared Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors.
6. **Hyperparameter Tuning:** Used cross-validation and GridSearchCV to tune candidate models.
7. **Threshold Tuning:** Selected a decision threshold of `0.40` to emphasize churn recall.
8. **Model Deployment:** Saved the fitted pipeline with Joblib and integrated it into a FastAPI application.
9. **Containerization and Hosting:** Packaged the application with Docker and deployed it on Render.

## Model Performance

The selected model is a tuned Logistic Regression pipeline with `class_weight="balanced"` and a decision threshold of `0.40`.

| Metric | Test-set result |
|---|---:|
| Recall | 86.6% |
| Precision | 46.7% |
| F1-score | 60.7% |
| True Positives | 324 |
| False Negatives | 50 |
| False Positives | 370 |
| True Negatives | 665 |

**Confusion matrix** (actual classes as rows; predicted classes as columns):

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 665 | 370 |
| Actual Churn | 50 | 324 |

The model identifies approximately 86.6% of actual churners in the held-out test set. The lower precision reflects the deliberate choice to flag more potential churners, which may be useful when missing a customer at risk is considered more costly than contacting a customer who would not churn.

These metrics are from the held-out test evaluation at the selected threshold, not a guarantee of performance on future customers.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Predict customer churn |

Interactive API documentation is available at `/docs` on the deployed service.

## Example Prediction Request

Send a POST request to `/predict` with customer details in JSON format:

```json
{
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
```

### Example Response

```json
{
  "churn_probability": 0.8503,
  "prediction": 1,
  "prediction_label": "Churn"
}
```

The probability shown is an example for the supplied customer input. A prediction of `1` indicates churn, while `0` indicates no churn.

## Project Structure

```text
customer_churn_prediction/
├── api/
│   ├── __init__.py
│   └── main.py
├── models/
│   └── churn_pipeline.joblib
├── src/
│   └── predict.py
├── data/                    # Local dataset; not tracked in Git
├── notebooks/               # Local analysis notebooks; not tracked in Git
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements-api.txt
└── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Gauresh2005/Customer-churn-prediction.git
cd Customer-churn-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install API dependencies

```bash
pip install -r requirements-api.txt
```

### 4. Start the API

```bash
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to test the prediction endpoint.

## Run with Docker

Build the image:

```bash
docker build -t churn-prediction-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 churn-prediction-api
```

Open `http://localhost:8000/docs` to test the API.

## Key Learnings

- Building an end-to-end machine learning workflow from EDA to deployment.
- Preventing data leakage through appropriate preprocessing and pipeline design.
- Comparing classification models and tuning hyperparameters using cross-validation.
- Understanding the trade-off between precision and recall when selecting a classification threshold.
- Saving and loading a trained machine learning pipeline.
- Creating and validating REST APIs with FastAPI and Pydantic.
- Containerizing a machine learning application using Docker.
- Deploying an API from GitHub to Render.

## Limitations and Future Improvements

- Add a React frontend for an interactive customer prediction experience.
- Monitor model performance and data drift as new customer data becomes available.
- Evaluate calibration and retrain the model when appropriate.
- Add automated tests and continuous integration.
- Compare against additional models and reassess the threshold using business costs and validation data.

## Author

**Gauresh Barge**

GitHub: https://github.com/Gauresh2005
