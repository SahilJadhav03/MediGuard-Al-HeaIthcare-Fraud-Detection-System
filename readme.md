# 🛡️ MediGuard AI — Healthcare Fraud Detection System

> AI-powered fraud detection for India's Ayushman Bharat PM-JAY health insurance scheme, aligned with IRDAI guidelines. Built with **LightGBM + FastAPI + React**.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start (One-Click)](#quick-start-one-click)
- [Manual Setup](#manual-setup)
- [Using the Dashboard](#using-the-dashboard)
- [API Reference](#api-reference)
- [Model Information](#model-information)
- [Dataset](#dataset)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

MediGuard AI detects fraudulent healthcare insurance claims at the **provider level** using machine learning. It processes CMS-format claims data (Inpatient, Outpatient, Beneficiary) and flags suspicious providers based on 30 engineered features.

**Key Results:**
- ✅ 100% Accuracy, Precision, Recall, F1-Score on test set
- ✅ ROC-AUC = 1.000 (5-fold cross-validation: 1.000 ± 0.000)
- ✅ 0 False Positives, 0 False Negatives on 1,082 test claims
- ✅ 23.91ms inference time for batch of 1,082 claims

---

## Features

| Feature | Description |
|---------|-------------|
| 📊 **Live Dashboard** | Real-time fraud stats, monthly trends, confusion matrix |
| 🔍 **Single Claim Predict** | Rule-based fraud scorer with 6 weighted risk components |
| 📂 **Batch Predict** | Upload CSV → get fraud probability for every provider |
| 📈 **Analytics** | Model comparison, feature importance, performance radar |
| 💰 **Savings Tracker** | Estimated INR savings from detected fraud |
| 📋 **Records** | Paginated view of 1,082 prediction records with filtering |
| 🔄 **Live Model Switcher** | Switch between 4 models without restarting the server |
| 🧪 **Test Cases** | 14 formal test cases for review rubric compliance |
| 📜 **Audit Log** | System event history |

---

## Tech Stack

**Backend**
- Python 3.13, FastAPI, Uvicorn
- LightGBM, XGBoost, Scikit-learn, SMOTENC (imbalanced-learn)
- Pandas, NumPy, Plotly, Matplotlib

**Frontend**
- React 19, Vite 7, Tailwind CSS 4
- Recharts (charts), Lucide React (icons), Axios (API calls)

---

## Project Structure

```
📁 Fradulant Health Detection/
│
├── 🦇 start.bat               ← ONE-CLICK LAUNCHER (double-click to run)
├── 📄 api_server.py           ← FastAPI backend (main server)
├── 📄 benchmark_models.py     ← Train & benchmark all 4 ML models
├── 📄 run_pipeline.py         ← Preprocess raw CSV → feature matrix
├── 📄 download_and_retrain.py ← Download fresh Kaggle data + retrain
├── 📄 generate_test_cases.py  ← Generate formal test cases
├── 📄 best_model.txt          ← Active model name (persists across restarts)
├── 📄 requirements.txt        ← Python dependencies
├── 📄 .gitignore              ← Git ignore rules
│
├── 📁 data/
│   ├── raw/                   ← Source CSVs (Beneficiary, Inpatient, Outpatient…)
│   └── processed/             ← X_train, X_test, y_train, y_test
│
├── 📁 docs/
│   └── healthcare_fraud_detection_prd.md   ← Product Requirements Document
│
├── 📁 models/                 ← Trained model PKL files
│   ├── lightgbm.pkl           ← Production model (active)
│   ├── xgboost.pkl
│   ├── random_forest.pkl
│   └── logistic_regression.pkl
│
├── 📁 model_benchmarks/       ← Per-model metrics, plots, predictions
│   ├── LightGBM/
│   ├── XGBoost/
│   ├── Random_Forest/
│   ├── Logistic_Regression/
│   ├── model_results.json
│   └── model_summary.csv
│
├── 📁 src/                    ← Core ML modules
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── project_utils.py
│
└── 📁 frontend/               ← React + Vite UI
    ├── src/App.jsx            ← Main application
    ├── package.json
    └── vite.config.js
```

---

## Prerequisites

Before running, install:

| Tool | Version | Download |
|------|---------|----------|
| **Python** | 3.10 or higher | [python.org](https://www.python.org/downloads/) ✔ tick "Add to PATH" |
| **Node.js** | 18 or higher (LTS) | [nodejs.org](https://nodejs.org/) |

> **Kaggle API** (only needed for `download_and_retrain.py`):
> Place your `kaggle.json` file in `C:\Users\<YourName>\.kaggle\` or set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables. Download from: Kaggle → Account → API → Create New Token.

---

## Quick Start (One-Click)

1. Open the project folder in File Explorer
2. **Double-click `start.bat`**

That's it. The script will:
1. ✅ Check Python and Node.js are installed
2. 📦 Install all Python packages (`pip install -r requirements.txt`)
3. 📦 Install all frontend packages (`npm install` in `frontend/`)
4. 🤖 Train models if not found (first run only, ~2-5 minutes)
5. 🚀 Open the **API server** at `http://localhost:8000` (cyan window)
6. 🌐 Open the **frontend UI** at `http://localhost:5177` (purple window)
7. 🖥️ Auto-open your browser at `http://localhost:5177`

---

## Manual Setup

If you prefer to run commands manually:

### Step 1 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Preprocess the raw data
```bash
python run_pipeline.py
```
This reads CSVs from `data/raw/`, engineers 30 provider-level features, and saves `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` in `data/processed/`.

### Step 3 — Train and benchmark all models
```bash
python benchmark_models.py
```
Trains LightGBM, XGBoost, Random Forest, and Logistic Regression with SMOTENC balancing. Saves models to `models/` and metrics to `model_benchmarks/`.

### Step 4 — Start the API backend
```bash
python api_server.py
```
Starts FastAPI on `http://localhost:8000`. Hot-reloads on file changes.

### Step 5 — Start the frontend
```bash
cd frontend
npm install        # first time only
npm run dev
```
Opens Vite dev server at `http://localhost:5177`. Open this URL in your browser.

### Optional — Download fresh Kaggle data and retrain
```bash
python download_and_retrain.py
```
Downloads the `rohitrox/healthcare-provider-fraud-detection-analysis` dataset from Kaggle, copies CSVs to `data/raw/`, reruns the pipeline, and retrains all models. Requires Kaggle credentials.

---

## Using the Dashboard

### Navigation (left sidebar)
| Page | Description |
|------|-------------|
| **Dashboard** | KPI cards (total claims, fraud detected, amount saved, fraud rate), monthly trend chart, confusion matrix |
| **Predict Fraud** | Two modes: Single Claim (rule-based scorer) and Batch CSV upload |
| **Analytics** | Model comparison table, feature importance bars, performance radar chart |
| **Savings** | Financial impact analysis, top-risk providers, monthly savings breakdown |
| **Records** | All 1,082 test-set prediction records with fraud/legit filter and pagination |
| **History** | System audit log of all events |
| **Settings** | Switch Model (live), Test Cases, Model Info, System Info |

### Single Claim Prediction
Fill in the form fields:
- **Claim Amount (₹)** — `> ₹2,00,000` = High Risk
- **Days Admitted** — `> 30 days` = Suspicious
- **Provider Fraud Rate** — `0.0` to `1.0` (biggest weight: 40%)
- **Chronic Conditions** — `0` with high claim = Upcoding risk
- **No. of Patients** — Large patient volume + high fraud rate = Red flag
- **No. of Physicians** — `> 8` unique physicians = Possible unbundling
- **Patient Age** — young patient + long stay = unusual

Use **🚨 Load Fraud Scenario** or **✅ Load Legit Scenario** to see examples instantly.

### Batch CSV Upload
Upload a CSV with the same column structure as the processed training data (30 features). The API returns fraud probability for every row.

### Switching Models (Settings → Switch Model)
Click **🔄 Switch to [Model Name]** to hot-swap the active model without restarting the server. The change persists across restarts (saved to `best_model.txt`).

---

## API Reference

Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Server health check |
| GET | `/api/dashboard/stats` | Dashboard KPIs and monthly trends |
| GET | `/api/analytics/overview` | Model comparison, feature importance |
| GET | `/api/records?page=1&limit=20` | Paginated prediction records |
| GET | `/api/savings` | Financial impact analysis |
| GET | `/api/history` | Audit log events |
| GET | `/api/test-cases` | Formal test case table |
| GET | `/api/available-models` | All 4 models with metrics |
| POST | `/api/switch-model` | Switch active model `{"model": "LightGBM"}` |
| POST | `/api/predict/single` | Single claim fraud score |
| POST | `/api/predict/batch` | Batch CSV fraud prediction |

Full interactive docs: `http://localhost:8000/docs`

---

## Model Information

All 4 models achieve **100% accuracy** on the test set. **LightGBM** is selected as the production model:

| Model | Accuracy | ROC-AUC | Train Time | Why |
|-------|----------|---------|-----------|-----|
| **LightGBM** ✅ | 100% | 1.000 | 61.6ms | Industry IRDAI standard, fast, explainable |
| XGBoost | 100% | 1.000 | 231ms | Powerful but slowest to train |
| Random Forest | 100% | 1.000 | 165ms | Robust but high memory usage |
| Logistic Regression | 100% | 1.000 | 24ms | Too simple for complex fraud patterns |

**Confusion Matrix (LightGBM on 1,082 test claims):**
```
              Predicted Legit   Predicted Fraud
Actual Legit       981 (TN)          0 (FP)
Actual Fraud         0 (FN)        101 (TP)
```

**Fraud Scorer — 6 Components (Single Claim):**
| Weight | Component | Threshold |
|--------|-----------|-----------|
| 40% | Provider Fraud Rate | 0% → clean, 80%+ → critical |
| 20% | Claim Amount Anomaly | > ₹1.25L avg → suspicious |
| 15% | Length of Stay | > 30 days → extreme (India avg: 6.2 days) |
| 10% | Upcoding Risk | Low conditions + high claim amount |
| 10% | Physician Spread | > 8 physicians → possible unbundling |
| 5%  | Deductible Ratio | Very low deductible vs claim |

---

## Dataset

- **Source:** CMS Healthcare Provider Fraud Detection (Kaggle: `rohitrox/healthcare-provider-fraud-detection-analysis`)
- **Context:** Aligned with IRDAI / Ayushman Bharat PM-JAY India health scheme
- **Raw files:** Beneficiary (`10.9MB`), Inpatient (`8.2MB`), Outpatient (`73.8MB`), Train labels (`0.1MB`)
- **Processed:** 5,418 total samples → 4,336 train / 1,082 test
- **Class balance:** 9.33% fraud → balanced using **SMOTENC** (405 → 3,923 fraud samples in train)
- **Features:** 30 provider-level aggregate features

---

## Troubleshooting

**`python` not recognized**
→ Reinstall Python from [python.org](https://python.org) and tick **"Add Python to PATH"** during installation.

**`node` not recognized**
→ Reinstall Node.js from [nodejs.org](https://nodejs.org) (choose LTS version).

**API starts but says "Loaded API Model: Random Forest" instead of LightGBM**
→ This was a bug fixed in the current version. If it still occurs, run:
```bash
echo LightGBM > best_model.txt
```
Then restart `api_server.py`.

**Frontend shows "Cannot connect to API"**
→ Make sure `python api_server.py` is running in a separate terminal before opening the frontend.

**Port 8000 already in use**
→ Another process is using the port. Find and stop it:
```bash
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Kaggle download fails**
→ Place `kaggle.json` in `C:\Users\<YourName>\.kaggle\`. Download from Kaggle → Account → API → Create New Token.

**pip install fails on LightGBM**
→ Install the Microsoft C++ Build Tools: [visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

---

*MediGuard AI — Built for IRDAI-aligned Ayushman Bharat PM-JAY fraud detection · Pan-India coverage · 28 States & 8 UTs*