import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartSales Dashboard",
    page_icon="",
    layout="wide"
)

# ── Styling ──────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .stApp { background-color: #0e1117; }
        .metric-card {
            background: linear-gradient(135deg, #1e2130, #2d3250);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #3d4466;
            text-align: center;
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #00d4aa;
        }
        .metric-label {
            font-size: 14px;
            color: #8892b0;
            margin-top: 4px;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# ── Header ───────────────────────────────────────────────────────────────────────
st.markdown("# SmartSales Forecasting Dashboard")
st.markdown("E-commerce revenue forecasting powered by XGBoost + MLflow")
st.divider()


# ── Helper functions ─────────────────────────────────────────────────────────────
def get_model_info():
    try:
        res = requests.get(f"{API_URL}/model-info", timeout=10)
        return res.json()
    except:
        return None


def get_forecast(days):
    try:
        res = requests.post(
            f"{API_URL}/forecast",
            json={"days": days},
            timeout=10
        )
        return res.json()
    except:
        return None


def get_single_prediction(date):
    try:
        res = requests.post(
            f"{API_URL}/predict",
            json={"date": date},
            timeout=10
        )
        return res.json()
    except:
        return None


def load_historical_data():
    try:
        df = pd.read_csv("./data/processed/sales_features.csv")
        df["date"] = pd.to_datetime(df["date"])
        return df.tail(90)  # Last 90 days
    except:
        return None


# ── Model Info Section ───────────────────────────────────────────────────────────
model_info = get_model_info()

if model_info:
    metrics = model_info.get("metrics", {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">£{metrics.get('mae', 0):,.0f}</div>
                <div class="metric-label">Mean Absolute Error</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">£{metrics.get('rmse', 0):,.0f}</div>
                <div class="metric-label">Root Mean Square Error</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('r2', 0):.3f}</div>
                <div class="metric-label">R² Score</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('mape', 0):.1f}%</div>
                <div class="metric-label">MAPE</div>
            </div>
        """, unsafe_allow_html=True)
else:
    st.error("Cannot connect to API. Make sure it is running!")
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)

# ── Two columns layout ───────────────────────────────────────────────────────────
left_col, right_col = st.columns([2, 1])

with left_col:
    # ── Forecast Chart ───────────────────────────────────────────────────────────
    st.markdown("### Revenue Forecast")

    forecast_days = st.slider(
        "Select forecast period (days)",
        min_value=7,
        max_value=30,
        value=14,
        step=7
    )

    forecast_data = get_forecast(forecast_days)
    historical_data = load_historical_data()

    if forecast_data and historical_data is not None:
        forecast_df = pd.DataFrame(forecast_data["forecast"])
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])

        fig = go.Figure()

        # Historical line
        fig.add_trace(go.Scatter(
            x=historical_data["date"],
            y=historical_data["total_revenue"],
            name="Historical Revenue",
            line=dict(color="#00d4aa", width=2),
            mode="lines"
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["predicted_revenue"],
            name="Forecasted Revenue",
            line=dict(color="#ff6b6b", width=2, dash="dash"),
            mode="lines+markers",
            marker=dict(size=6)
        ))

        fig.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white"),
            xaxis=dict(gridcolor="#2d3250", title="Date"),
            yaxis=dict(gridcolor="#2d3250", title="Revenue (£)"),
            legend=dict(
                bgcolor="#1e2130",
                bordercolor="#3d4466"
            ),
            hovermode="x unified",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Total forecast
        total = forecast_data["total_predicted_revenue"]
        st.success(f"Total Forecasted Revenue ({forecast_days} days): **£{total:,.2f}**")

with right_col:
    # ── Single Date Prediction ───────────────────────────────────────────────────
    st.markdown("### Predict Single Day")

    selected_date = st.date_input(
        "Select a date",
        value=datetime.now() + timedelta(days=1),
        min_value=datetime.now()
    )

    if st.button("Predict Revenue", use_container_width=True):
        result = get_single_prediction(str(selected_date))
        if result:
            revenue = result["predicted_revenue"]
            st.markdown(f"""
                <div class="metric-card" style="margin-top:16px">
                    <div class="metric-value">£{revenue:,.2f}</div>
                    <div class="metric-label">Predicted Revenue for {selected_date}</div>
                </div>
            """, unsafe_allow_html=True)

            # Context
            if revenue > 15000:
                st.success("High revenue day expected!")
            elif revenue > 8000:
                st.info("Average revenue day expected")
            else:
                st.warning("Low revenue day expected")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Forecast Table ───────────────────────────────────────────────────────────
    st.markdown("### Forecast Table")
    if forecast_data:
        forecast_df = pd.DataFrame(forecast_data["forecast"])
        forecast_df.columns = ["Date", "Predicted Revenue (£)"]
        forecast_df["Predicted Revenue (£)"] = forecast_df["Predicted Revenue (£)"].apply(
            lambda x: f"£{x:,.2f}"
        )
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

# ── Historical Revenue Chart ─────────────────────────────────────────────────────
st.divider()
st.markdown("### Historical Revenue (Last 90 Days)")

if historical_data is not None:
    fig2 = px.bar(
        historical_data,
        x="date",
        y="total_revenue",
        color="total_revenue",
        color_continuous_scale="teal",
        labels={"total_revenue": "Revenue (£)", "date": "Date"}
    )
    fig2.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#2d3250"),
        yaxis=dict(gridcolor="#2d3250"),
        coloraxis_showscale=False,
        height=300
    )
    st.plotly_chart(fig2, use_container_width=True)