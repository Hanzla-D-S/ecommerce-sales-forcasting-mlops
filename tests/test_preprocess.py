import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import clean_data, create_daily_sales
from src.features import add_time_features, add_lag_features, add_rolling_features


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_raw_data():
    """Create sample raw data for testing."""
    return pd.DataFrame({
        "Invoice": ["1001", "1002", "C1003", "1004", "1005"],
        "Customer ID": [1.0, 2.0, 3.0, None, 5.0],
        "Quantity": [10, 5, -3, 8, 2],
        "Price": [2.5, 10.0, 5.0, 0.0, 8.0],
        "InvoiceDate": pd.to_datetime([
            "2023-01-01", "2023-01-01",
            "2023-01-02", "2023-01-02", "2023-01-03"
        ])
    })


@pytest.fixture
def sample_daily_sales():
    """Create sample daily sales data for testing."""
    return pd.DataFrame({
        "date": pd.date_range(start="2023-01-01", periods=60, freq="D"),
        "total_revenue": np.random.uniform(1000, 5000, 60),
        "total_orders": np.random.randint(10, 100, 60),
        "total_items": np.random.randint(50, 500, 60),
        "unique_customers": np.random.randint(5, 50, 60)
    })


# ── Preprocessing Tests ──────────────────────────────────────────────────────────

def test_clean_data_removes_cancelled_orders(sample_raw_data):
    """Cancelled orders (Invoice starting with C) should be removed."""
    cleaned = clean_data(sample_raw_data)
    assert not any(cleaned["Invoice"].astype(str).str.startswith("C"))


def test_clean_data_removes_null_customers(sample_raw_data):
    """Rows with missing Customer ID should be removed."""
    cleaned = clean_data(sample_raw_data)
    assert cleaned["Customer ID"].isnull().sum() == 0


def test_clean_data_removes_negative_quantity(sample_raw_data):
    """Rows with negative quantity should be removed."""
    cleaned = clean_data(sample_raw_data)
    assert (cleaned["Quantity"] > 0).all()


def test_clean_data_removes_zero_price(sample_raw_data):
    """Rows with zero price should be removed."""
    cleaned = clean_data(sample_raw_data)
    assert (cleaned["Price"] > 0).all()


def test_clean_data_adds_revenue_column(sample_raw_data):
    """Revenue column should be added."""
    cleaned = clean_data(sample_raw_data)
    assert "Revenue" in cleaned.columns


def test_create_daily_sales_shape(sample_raw_data):
    """Daily sales should have correct columns."""
    cleaned = clean_data(sample_raw_data)
    daily = create_daily_sales(cleaned)
    assert "date" in daily.columns
    assert "total_revenue" in daily.columns
    assert "total_orders" in daily.columns


# ── Feature Engineering Tests ────────────────────────────────────────────────────

def test_add_time_features(sample_daily_sales):
    """Time features should be added correctly."""
    df = add_time_features(sample_daily_sales)
    assert "day_of_week" in df.columns
    assert "month" in df.columns
    assert "is_weekend" in df.columns
    assert df["is_weekend"].isin([0, 1]).all()


def test_add_lag_features(sample_daily_sales):
    """Lag features should be added."""
    df = add_lag_features(sample_daily_sales)
    assert "revenue_lag_1" in df.columns
    assert "revenue_lag_7" in df.columns


def test_add_rolling_features(sample_daily_sales):
    """Rolling features should be added."""
    df = add_rolling_features(sample_daily_sales)
    assert "revenue_rolling_7" in df.columns
    assert "revenue_rolling_30" in df.columns