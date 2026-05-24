# ⚡ Quick Setup Guide

## 🚀 5-Minute Setup

### Step 1: Environment Setup (2 min)
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Or activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Models (1 min)
Models should be in:
- `weights/detection/yolov8s.pt`
- `weights/detection/yolov8s-pose.pt`
- `weights/classification/lstm_best.pth`

### Step 3: Run! (Choose one - 2 min)

**Option A - Web Interface (RECOMMENDED)**
```bash
python run_web.py
# Opens at http://localhost:8501
```

**Option B - API Server**
```bash
python run_api.py
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

**Option C - Webcam Demo**
```bash
python run_webcam_demo.py
```

---

## 🔧 Troubleshooting

### Port Already in Use?
```bash
# Windows - Kill process on port 8501 (web) or 8000 (API)
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### No GPU Available?
- Streamlit web will work fine on CPU
- Processing will be slower but still functional
- For faster processing, ensure CUDA is installed

### Import Errors?
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or clean reinstall
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

---

## 📖 Full Documentation
See [README.md](README.md) for complete setup and API documentation.

---

## 🎯 Common Tasks

### Process Single Video
```bash
python test_pipeline.py path/to/video.mp4
```

### Batch Process Videos
1. Open web app: `python run_web.py`
2. Go to "Batch Processing"
3. Upload multiple videos
4. Click "▶️ Start Processing"

### View Analysis Results
1. Open web app: `python run_web.py`
2. Go to "Dashboard"
3. Check statistics and charts

### Export History
1. Go to "History" page
2. Click "📊 Export CSV" or "📄 Export JSON"

---

## 🐳 Docker Setup (Alternative)

```bash
# Build and run with Docker
docker-compose up -d

# Access web app at http://localhost:8501
# Access API at http://localhost:8000
```

---

**Stuck?** Check README.md for more detailed troubleshooting!
