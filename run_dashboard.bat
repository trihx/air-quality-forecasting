@echo off
REM Khởi động PM2.5 Forecasting Dashboard (Windows)

echo 🚀 Starting PM2.5 Forecasting Dashboard...
echo Vui long doi vai giay de Streamlit server khoi dong...

REM Kill tiến trình cũ nếu cổng 8501 đang bị chiếm (Windows)
FOR /F "tokens=5" %%T IN ('netstat -a -n -o ^| findstr :8501') DO (
    taskkill /F /PID %%T 2>NUL
)
uv run streamlit run app.py
pause
