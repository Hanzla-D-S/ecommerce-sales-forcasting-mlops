import pandas as pd
import numpy as np

PROCESSED_DATA_PATH = "./data/processed/sales_clean.csv"
FEATURES_DATA_PATH = "./data/processed/sales_features.csv"


def load_processed_data():
    """Load cleaned daily sales data."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_time_features(df):
    """Add time-based features."""
    print("Adding time features...")
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df


def add_lag_features(df):
    """Add lag features (past sales as features)."""
    print("Adding lag features...")
    df["revenue_lag_1"] = df["total_revenue"].shift(1)
    df["revenue_lag_7"] = df["total_revenue"].shift(7)
    df["revenue_lag_14"] = df["total_revenue"].shift(14)
    df["revenue_lag_30"] = df["total_revenue"].shift(30)
    return df


def add_rolling_features(df):
    """Add rolling average features."""
    print("Adding rolling features...")
    df["revenue_rolling_7"] = df["total_revenue"].shift(1).rolling(7).mean()
    df["revenue_rolling_14"] = df["total_revenue"].shift(1).rolling(14).mean()
    df["revenue_rolling_30"] = df["total_revenue"].shift(1).rolling(30).mean()
    return df


def save_features(df):
    """Save feature engineered data."""
    df = df.dropna()  # Drop rows with NaN from lag features
    df.to_csv(FEATURES_DATA_PATH, index=False)
    print(f"Features saved to '{FEATURES_DATA_PATH}'")
    print(f"Final shape: {df.shape}")


def run_feature_engineering():
    """Run full feature engineering pipeline."""
    print("Starting feature engineering...\n")
    df = load_processed_data()
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    save_features(df)
    print("\nFeature engineering complete!")
    print("\nFeatures created:")
    print([col for col in df.columns])
    return df


if __name__ == "__main__":
    run_feature_engineering()