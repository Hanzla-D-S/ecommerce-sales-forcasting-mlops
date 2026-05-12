import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import json

mlflow.set_tracking_uri("file:./mlruns")          
MLFLOW_EXPERIMENT = "smartsales-forecasting"

FEATURES_PATH = "./data/processed/sales_features.csv"
MODELS_PATH = "./models"
MLFLOW_EXPERIMENT = "smartsales-forecasting"

# Features to use for training
FEATURE_COLS = [
    "day_of_week", "day_of_month", "month", "quarter",
    "week_of_year", "is_weekend", "is_month_start", "is_month_end",
    "revenue_lag_1", "revenue_lag_7", "revenue_lag_14", "revenue_lag_30",
    "revenue_rolling_7", "revenue_rolling_14", "revenue_rolling_30",
    "total_orders", "total_items", "unique_customers"
]
TARGET_COL = "total_revenue"


def load_features():
    """Load feature engineered data."""
    df = pd.read_csv(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    print(f"Loaded features: {df.shape}")
    return df


def split_data(df):
    """Split data into train and test — last 30 days as test."""
    train = df[:-30]
    test = df[-30:]
    print(f"Train size: {len(train)} | Test size: {len(test)}")
    return train, test


def get_features_target(df):
    """Extract features and target."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def train_model(params=None):
    """Train XGBoost model with MLflow tracking."""

    # Default hyperparameters
    if params is None:
        params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }

    # Setup MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():

        print("Starting model training...")

        # Load and split data
        df = load_features()
        train, test = split_data(df)
        X_train, y_train = get_features_target(train)
        X_test, y_test = get_features_target(test)

        # Log parameters to MLflow
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("features", FEATURE_COLS)

        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Evaluate on test set
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test.values, y_pred)

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)

        print(f"\nModel Performance:")
        print(f"MAE:  £{metrics['mae']:,.2f}")
        print(f"RMSE: £{metrics['rmse']:,.2f}")
        print(f"R2:   {metrics['r2']:.4f}")
        print(f"MAPE: {metrics['mape']:.2f}%")

        # Save model with MLflow
        mlflow.xgboost.log_model(model, "xgboost_model")

        # Also save locally
        os.makedirs(MODELS_PATH, exist_ok=True)
        model.save_model(f"{MODELS_PATH}/xgboost_model.json")

        # Save metrics locally
        with open(f"{MODELS_PATH}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Log feature importance
        feature_importance = dict(
            zip(FEATURE_COLS, model.feature_importances_)
        )
        top_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        print(f"\nTop 5 Important Features:")
        for feat, imp in top_features:
            print(f"   {feat}: {imp:.4f}")

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow Run ID: {run_id}")
        print(f"Model saved to '{MODELS_PATH}/xgboost_model.json'")

    return model, metrics


if __name__ == "__main__":
    print("Starting SmartSales Model Training\n")

    # Experiment 1 — Default params
    print("=" * 50)
    print("Experiment 1: Default Parameters")
    print("=" * 50)
    model, metrics = train_model()

    # Experiment 2 — Different params (MLflow tracks both!)
    print("\n" + "=" * 50)
    print("Experiment 2: Tuned Parameters")
    print("=" * 50)
    tuned_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "random_state": 42
    }
    model2, metrics2 = train_model(params=tuned_params)

    print("\nTraining complete! Run 'mlflow ui' to see experiments!")