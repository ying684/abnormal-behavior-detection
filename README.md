# 🎥 Abnormal Behavior Detection System

A computer vision system for detecting abnormal behaviors (fighting, falling, loitering) in surveillance videos using YOLOv8, pose estimation, and LSTM classification.

## 🎯 Features

- **Real-time Detection**: Detect abnormal behaviors in video streams
- **Batch Processing**: Process multiple videos simultaneously
- **Web Interface**: User-friendly Streamlit dashboard
- **REST API**: FastAPI backend for programmatic access
- **Webcam Support**: Live camera feed analysis
- **Video Analysis**: Detailed behavior detection and timeline visualization

## 🏗️ System Architecture

```
├── api/                 # FastAPI backend
├── web/                 # Streamlit frontend
├── core/                # Core ML models and inference
│   ├── detection/      # YOLOv8 & pose estimation
│   ├── models/         # LSTM & Transformer models
│   ├── inference/      # Pipeline & postprocessing
│   └── training/       # Training utilities
├── config/             # Configuration files
└── scripts/            # Training & deployment scripts
```

## 📦 Requirements

- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)
- 4GB RAM minimum (8GB+ recommended)

## 🚀 Quick Start

### 1. Clone & Setup

```bash
# Clone repository
git clone <repository-url>
cd abnormal-behavior-detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Weights

The pre-trained weights should be in:
- `weights/detection/yolov8s.pt` - YOLOv8 detection model
- `weights/detection/yolov8s-pose.pt` - YOLOv8 pose model
- `weights/classification/lstm_best.pth` - LSTM classification model

### 3. Run Options

#### Option A: Web Interface (Recommended)
```bash
# Start the web application
python run_web.py

# Open browser at http://localhost:8501
```

#### Option B: API Server
```bash
# Start the API server
python run_api.py

# API available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

#### Option C: Webcam Demo
```bash
# Real-time camera feed analysis
python run_webcam_demo.py
```

#### Option D: Single Video Processing
```bash
python test_pipeline.py <video-path>
```

## 🌐 Web Interface Guide

### Pages Available

| Page | Function |
|------|----------|
| **Analyze Video** | Upload and analyze single video |
| **Batch Processing** | Process multiple videos at once |
| **Dashboard** | View statistics and analytics |
| **History** | Browse past analysis results |

### Quick Workflow
1. Go to "Analyze Video"
2. Upload your video file
3. Click "▶️ Analyze"
4. View results and download processed video
5. Check "Dashboard" for statistics

## 🔧 Configuration

Edit `config/settings.py` to customize:

```python
# Model parameters
input_size = 68           # Keypoint features
hidden_size = 256         # LSTM hidden dimension
num_classes = 4           # Behavior classes
seq_len = 20              # Sequence length

# Inference settings
confidence_threshold = 0.4
iou_threshold = 0.3

# API settings
host = "0.0.0.0"
port = 8000
```

## 📊 Detected Behaviors

- **🟢 Normal** - Regular human activity
- **🔴 Fighting** - Physical altercation
- **🟠 Falling** - Person falling down
- **🟣 Loitering** - Standing still for extended period

## 📁 Data Structure

```
data/
├── raw/           # Original video files
├── processed/     # Processed datasets
├── cache/         # Cache files
├── outputs/       # Generated analysis results
└── logs/          # System logs
```

## 🎓 Training Custom Models

### Train LSTM Model

```bash
python scripts/training/train_model.py \
    --data-dir data/processed \
    --output weights/classification \
    --epochs 50 \
    --batch-size 32
```

### Evaluate Model

```bash
python scripts/training/evaluate.py \
    --model weights/classification/lstm_best.pth \
    --test-dir data/processed/test
```

## 🐳 Docker Deployment

### Build Images
```bash
docker-compose build
```

### Start Services
```bash
docker-compose up -d
```

### Services
- **Web**: http://localhost:8501
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs

## 📝 API Endpoints

### Health Check
```bash
GET /api/health/
```

### Process Video
```bash
POST /api/video/process
Content-Type: multipart/form-data

file: <video-file>

Response:
{
    "success": true,
    "output_url": "/outputs/output_id.mp4",
    "info": {
        "duration": 30,
        "fps": 30.0,
        "resolution": "1920x1080",
        "frames": 900
    }
}
```

### Get Video Output
```bash
GET /outputs/output_{id}.mp4
```

## 🔍 Troubleshooting

### Issue: API not starting
```bash
# Check if port 8000 is in use
netstat -an | grep 8000
# Kill process using the port (Windows)
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Issue: CUDA not detected
```bash
# Install CPU version of PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Out of memory
- Reduce batch size in `config/settings.py`
- Process one video at a time
- Reduce video resolution

### Issue: Slow processing
- Enable GPU acceleration (install CUDA)
- Reduce input video resolution
- Skip frames during processing

## 📊 Performance

| Component | Processing Time (GPU) | Processing Time (CPU) |
|-----------|----------------------|----------------------|
| Detection + Pose | ~20ms/frame | ~100ms/frame |
| LSTM Classification | ~2ms/frame | ~10ms/frame |
| Full Pipeline | ~25ms/frame | ~120ms/frame |

## 📚 Project Structure

```
abnormal-behavior-detection/
├── api/                    # FastAPI application
│   ├── app.py             # Main app initialization
│   ├── routes/            # API endpoints
│   ├── models/            # Request/response models
│   └── services/          # Business logic
├── web/                    # Streamlit interface
│   ├── app.py             # Main web app
│   ├── pages/             # Page modules
│   └── components/        # UI components
├── core/                   # Core ML pipeline
│   ├── detection/         # Object detection
│   ├── models/            # Neural network models
│   ├── inference/         # Inference pipeline
│   └── training/          # Training modules
├── config/                # Configuration
├── scripts/               # Utility scripts
├── tests/                 # Unit tests
├── weights/               # Pre-trained models
├── data/                  # Data directory
├── docker/                # Docker configuration
├── notebooks/             # Jupyter notebooks
└── requirements.txt       # Python dependencies
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is part of CS338 Computer Vision course.

## 👥 Team

- Led by CS338 Course Instructors
- Developed for educational purposes

## 📧 Support

For issues and questions, please open a GitHub issue.

## 🔗 Links

- **Documentation**: See `docs/` folder
- **API Docs**: Run API and visit `/docs`
- **Training Guide**: See `docs/TRAINING.md`
- **API Reference**: See `docs/API.md`

---

**Last Updated**: May 2026
**Status**: ✅ Active
