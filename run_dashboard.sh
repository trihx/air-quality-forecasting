#!/bin/bash
# Khởi động PM2.5 Forecasting Dashboard (Mac/Linux)

echo "🚀 Starting PM2.5 Forecasting Dashboard..."
echo "Vui lòng đợi vài giây để Streamlit server khởi động..."

# Kill tiến trình cũ nếu cổng 8501 đang bị chiếm (tùy chọn)
lsof -ti :8501 | xargs kill -9 2>/dev/null

# Chạy Dashboard
uv run streamlit run app.py
