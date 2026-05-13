---
title: Ecommerce SmartSales
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

---
title: SmartSales Forecasting
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# SmartSales - E-commerce Sales Forecasting MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![MLflow](https://img.shields.io/badge/MLflow-2.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![CI](https://img.shields.io/badge/CI-GitHub_Actions-brightgreen)

A complete end-to-end MLOps pipeline for e-commerce sales forecasting. The system processes raw transaction data, trains an XGBoost model, tracks experiments with MLflow, versions data with DVC, serves predictions via FastAPI, and displays results on a Streamlit dashboard.

Live Demo: https://hanzlainam-ecommerce-sales-forcasting.hf.space

---

## Features

- Sales forecasting - predict revenue for any future date up to 30 days
- MLflow experiment tracking - compare runs, parameters, and metrics
- DVC data versioning - track data changes like code
- FastAPI REST API - clean endpoints for predictions and forecasts
- Streamlit dashboard - interactive charts and visualizations
- Docker containerization - runs anywhere with one command
- CI/CD pipeline - automated testing with GitHub Actions

---

## Architecture

```
Raw Data
    |
    v
preprocess.py  -  clean data, remove cancellations, aggregate daily sales
    |
    v
features.py  -  add lag features, rolling averages, time-based features
    |
    v
train.py  -  train XGBoost model, track experiments with MLflow
    |
    v
models/  -  saved xgboost_model.json and metrics.json
    |
    v
api/main.py  -  FastAPI serves /predict, /forecast, /model-info endpoints
    |
    v
dashboard/app.py  -  Streamlit displays charts and predictions
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | XGBoost |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| API | FastAPI |
| Dashboard | Streamlit + Plotly |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment | Hugging Face Spaces |

---

## Project Structure

```
smartsales/
├── src/
│   ├── preprocess.py        - clean and aggregate raw data
│   ├── features.py          - feature engineering
│   ├── train.py             - model training with MLflow tracking
│   └── predict.py           - load model and generate predictions
├── api/
│   └── main.py              - FastAPI prediction server
├── dashboard/
│   └── app.py               - Streamlit visualization dashboard
├── data/
│   ├── raw/                 - original dataset (DVC tracked)
│   └── processed/           - cleaned and feature engineered data
├── models/                  - saved model and metrics files
├── tests/
│   └── test_preprocess.py   - automated test suite
├── .github/workflows/
│   └── ci.yml               - GitHub Actions CI pipeline
├── Dockerfile               - Hugging Face deployment
├── Dockerfile.local         - local development
├── docker-compose.yml       - run all services together
├── start.sh                 - Hugging Face startup script
├── start.local.sh           - local Docker startup script
└── requirements.txt         - Python dependencies
```

---

## Run Locally

### Prerequisites
- Python 3.11 or higher
- Docker (optional)

### Without Docker

```bash
# Clone the repository
git clone https://github.com/Hanzla-D-S/ecommerce-sales-forcasting-mlops.git
cd ecommerce-sales-forcasting-mlops

# Create virtual environment
python -m venv env
env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python src/preprocess.py
python src/features.py

# Train the model
python src/train.py

# View MLflow experiments
mlflow ui

# Terminal 1 - Start API
uvicorn api.main:app --reload

# Terminal 2 - Start Dashboard
streamlit run dashboard/app.py
```

### With Docker

```bash
docker-compose up --build
```

Open http://localhost:8501 for the dashboard.
Open http://localhost:8000/docs for the API documentation.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Check API status |
| POST | /predict | Predict revenue for a single date |
| POST | /forecast | Forecast revenue for next N days |
| GET | /model-info | Get model type and performance metrics |

### Example Request

```json
POST /forecast
{
    "days": 7
}

Response:
{
    "forecast": [
        {"date": "2024-01-15", "predicted_revenue": 18432.50},
        {"date": "2024-01-16", "predicted_revenue": 21045.75}
    ],
    "total_predicted_revenue": 125678.90,
    "currency": "GBP"
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## CI/CD Pipeline

Every push to main branch automatically:
1. Sets up Python 3.11 environment
2. Installs all dependencies
3. Runs the full test suite
4. Builds the Docker image
5. Verifies the build succeeded

---

## Author

Muhammad Hanzla
- Email: hanzlainam204@gmail.com
- LinkedIn: [https://linkedin.com/in/YOUR_LINKEDIN](https://www.linkedin.com/in/muhammad-hanzla-data-science/)
- GitHub: https://github.com/Hanzla-D-S
