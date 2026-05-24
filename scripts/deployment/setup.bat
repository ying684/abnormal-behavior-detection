@echo off
echo Setting up Abnormal Behavior Detection System...
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not found!
    echo Please make sure you activated conda environment: conda activate cs338
    exit /b 1
)

echo Python found: 
python --version

REM Run Python setup script
python scripts\deployment\setup.py

pause