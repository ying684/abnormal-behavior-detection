# 🎥 Abnormal Behavior Detection System

Hệ thống nhận diện hành vi bất thường trong video theo thời gian thực, sử dụng YOLOv8 để phát hiện và theo dõi người, YOLOv8-Pose để ước lượng tư thế, và LSTM để phân loại hành vi.

---

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt môi trường & Dependencies](#cài-đặt-môi-trường--dependencies)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Huấn luyện mô hình](#huấn-luyện-mô-hình)
- [API Reference](#api-reference)
- [Giao diện Web](#giao-diện-web)

---

## Tổng quan

Hệ thống phân loại **4 loại hành vi**:

| Nhãn | Mô tả | Màu hiển thị |
|------|-------|--------------|
| `normal` | Hành vi bình thường | 🟢 Xanh lá |
| `fighting` | Đánh nhau | 🔴 Đỏ |
| `falling` | Ngã | 🟠 Cam |
| `loitering` | Đi lang thang bất thường | 🔵 Xanh dương |

**Pipeline xử lý:**

```
Video / Webcam
     ↓
YOLOv8 Tracker  →  Phát hiện & theo dõi người (track_id)
     ↓
YOLOv8-Pose     →  Ước lượng 17 keypoints tư thế
     ↓
Skeleton LSTM   →  Phân loại hành vi từ chuỗi keypoints
     ↓
Kết quả hiển thị (bounding box + nhãn + confidence)
```

---

## Kiến trúc hệ thống

```
project/
├── core/                         # Logic xử lý chính
│   ├── detection/
│   │   ├── tracker.py            # YOLOv8 person tracker
│   │   └── pose_estimator.py     # YOLOv8-Pose keypoint extractor
│   ├── models/
│   │   └── lstm_skeleton.py      # Bidirectional LSTM classifier
│   ├── inference/
│   │   └── pipeline.py           # Pipeline nhận diện realtime
│   └── training/
│       ├── dataset.py            # SkeletonDataset loader
│       └── trainer.py            # Training loop
├── api/                          # FastAPI backend
│   ├── app.py
│   ├── routes/
│   │   ├── health.py
│   │   └── video.py
│   └── services/
│       ├── recognition.py
│       └── storage.py
├── web/                          # Streamlit frontend
│   └── app.py
├── config/
│   └── settings.py               # Cấu hình toàn cục
├── weights/                      # Model weights (tự tạo)
│   ├── detection/
│   │   ├── yolov8s.pt
│   │   └── yolov8s-pose.pt
│   └── classification/
│       └── lstm_best.pth
└── data/                         # Dữ liệu (tự tạo)
    ├── raw/                      # Video gốc theo nhãn
    ├── processed/sequences/      # File .npy keypoint sequences
    ├── cache/                    # Upload tạm thời
    └── outputs/                  # Video đã xử lý
```

### Mô hình Skeleton LSTM

- **Input:** Chuỗi 20 frame, mỗi frame có 68 features (34 tọa độ xy + 34 velocity)
- **Architecture:** LayerNorm → Bidirectional LSTM (2 lớp, hidden=256) → FC (128) → Output (4 class)
- **Preprocessing:** Chuẩn hóa keypoints về trọng tâm, scale invariant, tính velocity giữa các frame

---

## Yêu cầu hệ thống

- Python 3.9+
- CUDA 11.8+ (khuyến nghị, có thể chạy trên CPU)
- RAM: tối thiểu 8GB (16GB khuyến nghị)
- GPU: NVIDIA 4GB VRAM+ (tùy chọn)

---

## Cài đặt

### Bước 1: Cài đặt môi trường ảo

Có 2 cách. **Khuyến nghị dùng Miniconda** vì conda tự quản lý cả CUDA toolkit và cuDNN — tránh được các lỗi driver phổ biến như `cudnn not found`, `CUDA version mismatch`, hay `OSError: libcublas.so not found` khi cài bằng pip thông thường.

---

#### Cách 1: `venv` (đơn giản, không cần cài thêm gì)

> Phù hợp nếu chỉ chạy CPU, hoặc đã cài sẵn CUDA driver và toolkit đúng phiên bản.

**Yêu cầu:** Python 3.9 – 3.11 đã được cài trên máy.

```bash
# 1. Tạo môi trường ảo trong thư mục dự án
python -m venv venv

# 2. Kích hoạt môi trường
# Windows (Command Prompt):
venv\Scripts\activate.bat

# Windows (PowerShell):
venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate

# 3. Kiểm tra đã vào đúng môi trường chưa
# Dấu nhắc lệnh sẽ hiện (venv) ở đầu dòng
python --version
```

**Cài PyTorch** (chọn đúng phiên bản CUDA của máy, kiểm tra bằng `nvidia-smi`):

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Không có GPU / chỉ dùng CPU
pip install torch torchvision torchaudio
```

**Cài các thư viện còn lại:**

```bash
pip install -r requirements.txt
```

**Thoát môi trường:**

```bash
deactivate
```

---

#### Cách 2: `Miniconda` ⭐ (khuyến nghị — ổn định hơn với GPU)

> Conda tự cài CUDA toolkit và cuDNN phù hợp vào môi trường ảo, không cần lo về việc driver hệ thống có khớp không.

**Bước 2.1 — Cài Miniconda** (nếu chưa có):

- Tải installer tại: https://docs.conda.io/en/latest/miniconda.html
- Chọn đúng hệ điều hành (Windows / Linux / macOS)

```bash
# Linux — cài nhanh qua terminal:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Làm theo hướng dẫn, chọn "yes" khi hỏi init conda

# Sau khi cài xong, khởi động lại terminal hoặc chạy:
source ~/.bashrc
```

Kiểm tra đã cài xong:

```bash
conda --version
```

**Bước 2.2 — Tạo môi trường:**

```bash
conda create -n behavior_detection python=3.10 -y
```

**Bước 2.3 — Kích hoạt môi trường:**

```bash
conda activate behavior_detection

# Dấu nhắc lệnh sẽ hiện (behavior_detection) ở đầu dòng
python --version
```

**Bước 2.4 — Cài PyTorch**

> Không chắc máy dùng CUDA bao nhiêu? Chạy `nvidia-smi` và nhìn cột `CUDA Version` góc trên bên phải.

Có 3 cách, thử theo thứ tự ưu tiên từ trên xuống:

**Cách ưu tiên 1 — `conda install` (khuyến nghị, conda tự lo CUDA + cuDNN):**

> Không chắc máy dùng CUDA bao nhiêu? Chạy `nvidia-smi` và nhìn cột `CUDA Version` góc trên bên phải.

```bash
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# CUDA 12.4
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Không có GPU
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

Nếu lệnh trên bị lỗi solver hoặc treo lâu, thử thêm flag libmamba:

```bash
conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba
# rồi chạy lại lệnh conda install ở trên
```

---

**Cách ưu tiên 2 — `pip install` đơn giản (thường tự nhận CUDA trong môi trường conda):**

Trong môi trường conda, pip cũng có thể tự kéo đúng bản torch có CUDA mà không cần thêm gì:

```bash
pip install torch
```

Sau đó kiểm tra GPU ngay (xem bên dưới). Nếu `torch.cuda.is_available()` trả về `True` thì dừng ở đây, không cần làm thêm gì.

---

**Cách ưu tiên 3 — `pip install` với index URL (chỉ định thẳng phiên bản CUDA):**

Dùng khi 2 cách trên cài xong nhưng torch vẫn không nhận GPU:

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Không có GPU / chỉ dùng CPU
pip install torch torchvision torchaudio
```

---

**Bước 2.5 — Cài các thư viện còn lại:**

```bash
pip install -r requirements.txt
```

**Thoát môi trường:**

```bash
conda deactivate
```

---

**Kiểm tra PyTorch đã nhận GPU chưa** (chạy sau khi cài xong, dù dùng cách nào):

```python
import torch
print("CUDA available:", torch.cuda.is_available())   # True nếu có GPU
print("CUDA version  :", torch.version.cuda)
print("GPU name      :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
```

---

### Bước 2: Setup dự án

```bash
# Tạo thư mục, tải YOLO weights tự động
python scripts/deployment/setup.py
```

Script sẽ tự động:
- Tạo toàn bộ thư mục cần thiết (`weights/`, `data/`, ...)
- Tải YOLOv8 và YOLOv8-Pose weights về `weights/detection/`
- Kiểm tra LSTM weights

### Bước 3: Tải YOLO weights thủ công (nếu bước trên lỗi)

```bash
python down_model_yolo.py
```

---

## Cấu trúc thư mục

Sau khi setup, thư mục `weights/` cần có:

```
weights/
├── detection/
│   ├── yolov8s.pt          # YOLOv8 detection
│   └── yolov8s-pose.pt     # YOLOv8 pose estimation
└── classification/
    └── lstm_best.pth       # LSTM model đã train
```

Thư mục `data/raw/` cần có cấu trúc theo nhãn (để train):

```
data/raw/
├── normal/       # *.mp4 video hành vi bình thường
├── fighting/     # *.mp4 video đánh nhau
├── falling/      # *.mp4 video ngã
└── loitering/    # *.mp4 video đi lang thang
```

---

## Hướng dẫn sử dụng

### Demo Webcam realtime

```bash
python run_webcam_demo.py
```

Nhấn `ESC` để thoát.

### Demo với file video

```python
from core.inference.pipeline import RealtimeBehaviorRecognizer
from config.settings import settings

recognizer = RealtimeBehaviorRecognizer(
    yolo_weight="weights/detection/yolov8s.pt",
    pose_weight="weights/detection/yolov8s-pose.pt",
    lstm_weight="weights/classification/lstm_best.pth",
    classes=["normal", "fighting", "falling", "loitering"],
    device="cuda",
    seq_len=20,
    min_frames_for_decision=10
)

recognizer.recognize_from_video(
    source="video.mp4",
    display=True,
    save_output="output.mp4"
)
```

### Kiểm tra toàn bộ pipeline

```bash
python test_pipeline.py
```

---

## Huấn luyện mô hình

### Bước 1: Chuẩn bị dataset

Đặt video vào `data/raw/<class>/` rồi chạy:

```bash
python scripts/data_processing/build_sequences.py
```

Script sẽ trích xuất keypoint sequences và lưu file `.npy` vào `data/processed/sequences/<class>/`.

### Bước 2: Train mô hình

```bash
python scripts/training/train_model.py
```

Model tốt nhất sẽ được lưu tự động vào `weights/classification/lstm_best.pth`.

**Cấu hình training mặc định** (trong `config/settings.py`):

| Tham số | Giá trị |
|---------|---------|
| Batch size | 32 |
| Learning rate | 1e-3 |
| Epochs | 50 |
| Early stopping | 10 epochs |
| Seq length | 20 frames |
| Hidden size LSTM | 256 |

---

## API Reference

### Khởi động API server

```bash
python run_api.py
# hoặc
cd api && python app.py
```

API chạy tại `http://localhost:8000`. Swagger docs tại `http://localhost:8000/docs`.

### Endpoints

#### `GET /api/health/`
Kiểm tra trạng thái API.

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00",
  "service": "Behavior Recognition API"
}
```

#### `POST /api/video/process`
Upload và xử lý một video.

- **Form data:** `file` (mp4, avi, mov, mkv, tối đa 200MB)
- **Response:**

```json
{
  "success": true,
  "upload_id": "uuid",
  "output_url": "/outputs/output_uuid.mp4",
  "info": {
    "fps": 30.0,
    "frames": 900,
    "width": 1280,
    "height": 720,
    "duration": 30.0,
    "resolution": "1280x720"
  }
}
```

#### `POST /api/video/batch-process`
Xử lý nhiều video cùng lúc.

- **Form data:** `files[]` (danh sách video)

#### `GET /api/video/download/{file_id}`
Tải về video đã xử lý.

#### `GET /api/video/list-outputs`
Liệt kê tất cả video đầu ra.

---

## Giao diện Web

### Khởi động Web

```bash
python run_web.py
# hoặc
cd web && streamlit run app.py
```

Web chạy tại `http://localhost:8501`.

### Các tính năng

| Trang | Mô tả |
|-------|-------|
| **Analyze Video** | Upload và phân tích đơn video, xem kết quả, tải về |
| **Batch Processing** | Xử lý hàng loạt video, tải kết quả dạng ZIP |
| **Dashboard** | Biểu đồ thống kê các video đã xử lý |
| **History** | Lịch sử xử lý, export CSV |

> **Lưu ý:** API server phải đang chạy trước khi dùng Web interface.

---

## Kiểm thử

```bash
# Test pipeline
python test_pipeline.py

# Test API (cần API đang chạy)
python test_api.py
```

---

## Cấu hình

Toàn bộ cấu hình tập trung tại `config/settings.py`. Các tham số chính:

```python
# Đường dẫn weights
yolo_weight   = "weights/detection/yolov8s.pt"
pose_weight   = "weights/detection/yolov8s-pose.pt"
lstm_weight   = "weights/classification/lstm_best.pth"

# Inference
seq_len                 = 20    # Độ dài chuỗi frame
min_frames_for_decision = 10    # Số frame tối thiểu để dự đoán
confidence_threshold    = 0.4

# API
host = "0.0.0.0"
port = 8000
max_upload_size = 200MB
```

---

## Môn học

CS338 - Nhận Dạng
