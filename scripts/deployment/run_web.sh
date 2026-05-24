#!/bin/bash
# scripts/deployment/run_web.sh

echo "Starting Web interface..."

# Check Python
if ! command -v python &> /dev/null; then
    echo "Python is not installed!"
    exit 1
fi

# Start Web
cd web
streamlit run app.py
