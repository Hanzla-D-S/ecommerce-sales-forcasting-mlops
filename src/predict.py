import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
from datetime import datetime, timedelta

MODEL_PATH = "./models/xgboost_model.json"
FEATURES_PATH = "./data/processed/sales_features.csv"
METRICS_PATH = "./models/metrics.json"

FEATURE_COLS = [
    "day_of_week", "day_of_month", "month", "quarter",
    "week_of_year", "is_weekend", "is_month_start", "is_month_end",
    "revenue_lag_1", "revenue_lag_7", "revenue_lag_14", "revenue_lag_30",
    "revenue_rolling_7", "revenue_rolling_14", "revenue_rolling_30",
    "total_orders", "total_items", "unique_customers"
]


def load_model():
    """Load trained XGBoost model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found! Please run 'python src/train.py' first."
        )
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


def load_metrics():
    """Load model metrics."""
    if not os.path.exists(METRICS_PATH):
        return {}
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def get_latest_data():
    """Get latest data for lag features."""
    df = pd.read_csv(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def build_features_for_date(target_date: datetime, df: pd.DataFrame):
    """Build feature row for a given date."""
    target_date = pd.Timestamp(target_date)

    # Time features
    features = {
        "day_of_week": target_date.dayofweek,
        "day_of_month": target_date.day,
        "month": target_date.month,
        "quarter": target_date.quarter,
        "week_of_year": target_date.isocalendar().week,
        "is_weekend": int(target_date.dayofweek in [5, 6]),
        "is_month_start": int(target_date.is_month_start),
        "is_month_end": int(target_date.is_month_end),
    }

    # Lag features from historical data
    last_row = df.iloc[-1]
    features["revenue_lag_1"] = df["total_revenue"].iloc[-1]
    features["revenue_lag_7"] = df["total_revenue"].iloc[-7] if len(df) >= 7 else last_row["total_revenue"]
    features["revenue_lag_14"] = df["total_revenue"].iloc[-14] if len(df) >= 14 else last_row["total_revenue"]
    features["revenue_lag_30"] = df["total_revenue"].iloc[-30] if len(df) >= 30 else last_row["total_revenue"]

    # Rolling features
    features["revenue_rolling_7"] = df["total_revenue"].iloc[-7:].mean()
    features["revenue_rolling_14"] = df["total_revenue"].iloc[-14:].mean()
    features["revenue_rolling_30"] = df["total_revenue"].iloc[-30:].mean()

    # Other features — use recent averages
    features["total_orders"] = df["total_orders"].iloc[-7:].mean()
    features["total_items"] = df["total_items"].iloc[-7:].mean()
    features["unique_customers"] = df["unique_customers"].iloc[-7:].mean()

    return features


def predict_single(target_date: datetime):
    """Predict revenue for a single date."""
    model = load_model()
    df = get_latest_data()
    features = build_features_for_date(target_date, df)
    X = pd.DataFrame([features])[FEATURE_COLS]
    # print(X)
    prediction = float(model.predict(X)[0])
    return max(0, prediction)  # Revenue can't be negative


def predict_next_n_days(n: int = 7):
    """Predict revenue for next N days."""
    model = load_model()
    df = get_latest_data()
    last_date = df["date"].max()
    predictions = []

    for i in range(1, n + 1):
        target_date = last_date + timedelta(days=i)
        features = build_features_for_date(target_date, df)
        X = pd.DataFrame([features])[FEATURE_COLS]
        pred = float(model.predict(X)[0])
        predictions.append({
            "date": target_date.strftime("%Y-%m-%d"),
            "predicted_revenue": round(max(0, pred), 2)
        })

    return predictions


# Global model instance
model = load_model()
metrics = load_metrics()