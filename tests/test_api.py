# test_api.py
"""Quick test for API endpoints"""

import requests
import time

def test_api():
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/api/health/")
        if response.status_code == 200:
            print("✅ API is healthy:", response.json())
        else:
            print("❌ Health check failed")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Make sure API is running: cd api && python app.py")
        return False
    
    return True

if __name__ == "__main__":
    test_api()
