# Product Requirements Document (PRD)

# Healthcare Fraud Detection System

## 1. Product Overview

**Product Name:** Healthcare Fraud Detection System (HFDS)

AI-powered system designed to detect fraudulent healthcare insurance
claims using ensemble machine learning models with a modern web
dashboard.

The system analyzes healthcare claim datasets and predicts fraudulent
activity in real time through a React frontend and FastAPI backend.

------------------------------------------------------------------------

## 2. Problem Statement

Healthcare insurance systems process massive volumes of claims.
Detecting fraud manually is difficult due to:

-   Large datasets
-   Highly imbalanced fraud data
-   Complex claim relationships
-   Mixed structured features

Fraudulent claims lead to billions in losses globally. This system aims
to automate fraud detection using machine learning.

------------------------------------------------------------------------

## 3. Objectives

1.  Handle imbalanced healthcare datasets using SMOTENC.
2.  Train ensemble machine learning models.
3.  Provide real-time fraud detection.
4.  Build an intuitive fraud detection dashboard.
5.  Improve fraud detection accuracy.

------------------------------------------------------------------------

## 4. Team Members

-   Jai Sonar (Roll No. 56)
-   Prekshit Sonawane (Roll No. 60)
-   Tirtha Sonawane (Roll No. 61)
-   Akansha Tingase (Roll No. 64)

Guide: Prof. Atul Chaudhari

------------------------------------------------------------------------

## 5. System Architecture

High level architecture:

User → React Frontend → FastAPI Backend → ML Model → Prediction Output

------------------------------------------------------------------------

## 6. Technology Stack

### Frontend

-   React
-   Vite
-   Tailwind CSS v4
-   Shadcn UI
-   Framer Motion
-   Lottie Animations

### Backend

-   FastAPI
-   Python
-   Uvicorn

### Machine Learning

-   Scikit-learn
-   XGBoost
-   LightGBM
-   Random Forest
-   Logistic Regression

### Data Processing

-   Pandas
-   NumPy
-   SMOTENC (Imbalanced-Learn)

------------------------------------------------------------------------

## 7. Dataset

Dataset Source: Kaggle Healthcare Provider Fraud Detection

Dataset files:

-   Train.csv
-   Beneficiary.csv
-   Inpatient.csv
-   Outpatient.csv

Contains: - 500k+ claims - 5000+ providers - 100+ engineered features

------------------------------------------------------------------------

## 8. Data Processing Pipeline

Data Loading → Cleaning → Feature Engineering → Aggregation → SMOTENC
balancing → Encoding → Scaling → Train/Test Split

Outputs: - X_train - X_test - y_train - y_test

------------------------------------------------------------------------

## 9. Machine Learning Pipeline

Models benchmarked:

-   Random Forest
-   Gradient Boosting
-   XGBoost
-   LightGBM
-   Logistic Regression

Evaluation metrics:

-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   ROC-AUC

Best model automatically saved as:

best_model.pkl

------------------------------------------------------------------------

## 10. Fraud Detection Workflow

User uploads CSV → Frontend sends file to API → Backend preprocesses
data → Model predicts fraud probability → Results returned to dashboard.

Example Output:

Fraud Probability: 0.83 Prediction: Fraudulent Claim

------------------------------------------------------------------------

## 11. API Endpoint

POST /predict

Request: multipart/form-data file: claims.csv

Response:

{ "status": "success", "predictions": \[ { "claim_id": 1234,
"fraud_probability": 0.83, "prediction": "fraud" } \] }

------------------------------------------------------------------------

## 12. Dashboard Features

-   Dataset upload
-   Fraud analysis
-   AI processing animation
-   Results visualization
-   Fraud probability charts

------------------------------------------------------------------------

## 13. Deployment

Backend:

python api_server.py

Frontend:

npm install npm run dev

Backend runs on: http://localhost:8000

Frontend runs on: http://localhost:5177

------------------------------------------------------------------------

## 14. Future Enhancements

### Document Fraud Detection

Support for: - PDF claim documents - Medical bill images

Pipeline: PDF/Image → OCR → Text Extraction → Fraud Model

### Explainable AI

Add SHAP explanations for fraud predictions.

### Graph-based Fraud Detection

Detect fraud rings using network analysis.

------------------------------------------------------------------------

## 15. Success Metrics

-   High fraud recall
-   Low false positives
-   Fast inference time
-   Dashboard usability

------------------------------------------------------------------------

## 16. Conclusion

The Healthcare Fraud Detection System combines ensemble machine learning
with a modern web dashboard to identify fraudulent healthcare insurance
claims efficiently.

The system enables automated fraud analysis, scalable API architecture,
and real-time decision support for insurance fraud detection.
