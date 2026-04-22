@echo off
title SACT Intelligent Scheduler
echo ============================================
echo   SACT Intelligent Scheduling System
echo   Velindre Cancer Centre
echo ============================================
echo.
echo Starting application...
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found.
    echo Please run: python -m venv venv
    echo Then: pip install -r requirements.txt
    echo.
)

REM Run the Streamlit application
streamlit run app.py --server.port 8501 --server.headless false

pause
