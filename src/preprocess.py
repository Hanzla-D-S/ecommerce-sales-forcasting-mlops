import pandas as pd
import numpy as np
import os
import datetime as dt

RAW_DATA_PATH = "./data/raw/online_retail_II.csv"
PROCESSED_DATA_PATH = "./data/processed/sales_clean.csv"


def load_data():
    print("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH, encoding="ISO-8859-1")
    print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns")
    return df


def clean_data(df):
    """Clean and preprocess the dataset."""
    print("Cleaning data...")

    # Drop rows with missing Customer ID
    df = df.dropna(subset=["Customer ID"])

    # Drop cancelled orders (Invoice starting with C)
    df = df[~df["Invoice"].astype(str).str.startswith("C")]

    # Drop rows with negative or zero quantity/price
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]

    # Convert InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Add revenue column
    df["Revenue"] = df["Quantity"] * df["Price"]

    # Remove outliers (top 1% revenue)
    upper_limit = df["Revenue"].quantile(0.99)
    df = df[df["Revenue"] <= upper_limit]

    print(f"Clean data shape: {df.shape[0]:,} rows")
    return df


def create_daily_sales(df):
    """Aggregate data into daily sales."""
    print("Creating daily sales aggregation...")

    daily_sales = df.groupby(
        df["InvoiceDate"].dt.date
    ).agg(
        total_revenue=("Revenue", "sum"),
        total_orders=("Invoice", "nunique"),
        total_items=("Quantity", "sum"),
        unique_customers=("Customer ID", "nunique")
    ).reset_index()

    daily_sales.rename(columns={"InvoiceDate": "date"}, inplace=True)
    daily_sales["date"] = pd.to_datetime(daily_sales["date"])
    daily_sales = daily_sales.sort_values("date")

    print(f"Daily sales shape: {daily_sales.shape}")
    return daily_sales


def save_processed_data(df):
    """Save cleaned data to processed folder."""
    os.makedirs("./data/processed", exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed data to '{PROCESSED_DATA_PATH}'")


def run_pipeline():
    """Run full preprocessing pipeline."""
    print("preprocessing pipeline...\n")
    df = load_data()
    df = clean_data(df)
    daily_sales = create_daily_sales(df)
    save_processed_data(daily_sales)
    print("\nPreprocessing complete!")
    print(f"\nPreview:")
    print(daily_sales.head())
    return daily_sales


if __name__ == "__main__":
    run_pipeline()