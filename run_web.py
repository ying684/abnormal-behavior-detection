# run_web.py
"""Run Streamlit web app from root directory"""
import sys
import os
from pathlib import Path
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    print("Starting Web interface...")
    
    # Change to web directory
    web_dir = Path(__file__).parent / "web"
    os.chdir(web_dir)
    
    # Run streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
