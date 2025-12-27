@echo off
REM Privacy-Fraud-AI Setup Script for Windows
REM Run this to set up the project locally

echo.
echo ============================================
echo Privacy-Fraud-AI Local Setup
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.13+
    pause
    exit /b 1
)

echo [OK] Python found
echo.

REM Create virtual environment
echo Creating virtual environment...
if not exist venv (
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo [OK] Virtual environment activated
echo.

REM Install requirements
echo Installing dependencies (this may take 2-3 minutes)...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

echo [OK] Dependencies installed
echo.

REM Create directories
if not exist data mkdir data
if not exist models mkdir models
echo [OK] Data and models directories created

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo To run the application:
echo   streamlit run notebooks/ui.py
echo.
echo Then open in browser:
echo   http://localhost:8501
echo.
pause
