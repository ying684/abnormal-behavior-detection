@echo off
echo Starting Abnormal Behavior Detection System...
echo.

REM Start API in new window
start "API Server" cmd /k "cd api && python app.py"

REM Wait a bit for API to start
timeout /t 5 /nobreak >nul

REM Start Web in new window
start "Web Interface" cmd /k "cd web && streamlit run app.py"

echo.
echo System is starting...
echo API: http://localhost:8000
echo Web: http://localhost:8501
echo.
echo Press any key to stop all services...
pause >nul

REM Kill Python processes (careful with this!)
taskkill /F /IM python.exe