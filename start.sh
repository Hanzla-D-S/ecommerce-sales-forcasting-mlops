#!/bin/bash

echo "🚀 Training model..."
python src/train.py

echo "🚀 Starting FastAPI backend..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

echo "⏳ Waiting for backend..."
sleep 15

echo "🚀 Starting Streamlit dashboard..."
streamlit run dashboard/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false