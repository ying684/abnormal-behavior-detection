#!/bin/bash
# scripts/deployment/run_api.sh

echo "Starting API server..."

# Check Python
if ! command -v python &> /dev/null; then
    echo "Python is not installed!"
    exit 1
fi

# Start API
cd api
python app.py
