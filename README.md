# 🎥 Real-time Abnormal Behavior Detection System

Hệ thống Camera giám sát thông minh nhận diện hành vi bất thường theo thời gian thực.  
Sử dụng kiến trúc lai (Hybrid Architecture) kết hợp **YOLOv8-Pose**, mô hình **1D-CNN + BiLSTM** siêu nhẹ và thuật toán **Rule-based** để đạt tốc độ FPS cao mà không cần API Backend.

---

## 📋 Mục lục
- [Tổng quan](#tổng-quan)
- [Cơ chế nhận diện (Hybrid)](#cơ-chế-nhận-diện-hybrid)
- [Kiến trúc thư mục](#kiến-trúc-thư-mục)
- [Mô hình Skeleton LSTM](#mô-hình-skeleton-lstm)
- [Cài đặt môi trường](#cài-đặt-môi-trường)
- [Hướng dẫn khởi chạy](#hướng-dẫn-khởi-chạy)
- [Huấn luyện lại Model](#huấn-luyện-lại-model)
- [Pipeline minh họa](#pipeline-minh-họa)

---

## Tổng quan

Hệ thống phân loại và cảnh báo **4 trạng thái hành vi**:

| Nhãn       | Phương pháp xử lý | Màu cảnh báo |
|------------|-------------------|--------------|
| `normal`   | AI (LSTM)         | 🟢 Xanh lá   |
| `fighting` | AI (LSTM)         | 🔴 Đỏ        |
| `falling`  | AI (LSTM)         | 🟠 Cam       |
| `loitering`| Rule-based        | 🔵 Xanh dương|

**Pipeline luồng dữ liệu (Real-time):**
1. **Pose Tracking:** YOLOv8-Pose trích xuất Bounding Box và 17 Keypoints, gán ID qua BoT-SORT.
2. **Tiền xử lý:** Chuẩn hóa không gian (trừ tâm) + trích xuất vận tốc → Tensor (30 frames, 68 features).
3. **AI Inference:** Đưa vào mạng 1D-CNN + BiLSTM để suy luận hành động (Normal/Fight/Fall).
4. **Hậu xử lý:** Majority Voting + tính toán thời gian đứng hình để kích hoạt Loitering.
5. **Dashboard:** Ghi Log CSV để Web Streamlit cập nhật biểu đồ mỗi 3 giây.

---

## Cơ chế nhận diện (Hybrid)

- **YOLOv8-Pose:** Trích xuất khung xương người.
- **1D-CNN + BiLSTM:** Phân loại hành vi động (Fight/Fall/Normal).
- **Rule-based:** Phát hiện hành vi tĩnh (Loitering).

---

## Kiến trúc thư mục

```text
project/
├── core/                         
│   ├── detection/pose_tracker.py       # YOLOv8 Pose + BoT-SORT
│   ├── inference/
│   │   ├── pipeline.py                 # Luồng nhận diện Real-time
│   │   ├── preprocessor.py             # Chuẩn hóa & vận tốc
│   │   └── postprocessor.py            # Voting & Threshold
│   ├── models/lstm_skeleton.py         # 1D-CNN + BiLSTM
│   └── training/
│       ├── dataset.py                  # DataLoader & Augmentation
│       └── evaluation.py               # Metrics
├── web/                                # Giao diện Web (Streamlit)
│   ├── app.py
│   └── pages/analysis.py
├── config/settings.py                  # Cấu hình toàn cục
├── scripts/
│   ├── data_processing/build_sequences.py
│   └── training/train_model.py
├── data/                               # Data & Logs
├── weights/                            # Weights (.pt, .pth)
├── run_web.py                          # Khởi động Dashboard
└── run_webcam_demo.py                  # Khởi động Camera AI
