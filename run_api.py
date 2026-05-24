# run_api.py
"""Run API server from root directory"""
import sys
import os
from pathlib import Path

# Change to API directory
os.chdir(Path(__file__).parent / "api")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run
from api.app import app
import uvicorn

if __name__ == "__main__":
    from config.settings import settings
    
    print("Starting API server...")
    print(f"Host: {settings.api.host}")
    print(f"Port: {settings.api.port}")
    print(f"API URL: http://localhost:{settings.api.port}")
    print(f"Docs: http://localhost:{settings.api.port}/docs")
    
    # Use string import path when reload is enabled
    if settings.api.reload:
        uvicorn.run(
            "api.app:app",
            host=settings.api.host,
            port=settings.api.port,
            reload=True
        )
    else:
        uvicorn.run(
            app,
            host=settings.api.host,
            port=settings.api.port,
            reload=False
        )
