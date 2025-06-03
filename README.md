# Hệ thống Nhận diện Người & Khuôn Mặt

## 1. Tổng quan hệ thống

Hệ thống Nhận diện Người & Khuôn Mặt là ứng dụng web thời gian thực sử dụng các mô hình YOLO (You Only Look Once) và TensorFlow để phát hiện người, khuôn mặt và nhận diện cảm xúc trên khuôn mặt từ camera. Hệ thống được thiết kế để hoạt động trên cả máy tính để bàn và thiết bị di động với giao diện thích ứng.

### 1.1. Chức năng chính

- Phát hiện người trong khung hình video theo thời gian thực
- Phát hiện khuôn mặt trong các vùng chứa người
- Nhận diện cảm xúc trên khuôn mặt (7 loại cảm xúc cơ bản)
- Hiển thị thống kê về số lượng người, khuôn mặt và phân bố cảm xúc
- Tùy chỉnh hiển thị các phần tử trên giao diện

### 1.2. Công nghệ sử dụng

- **Backend**: FastAPI (Python), YOLO, TensorFlow
- **Frontend**: HTML, CSS, JavaScript (vanilla)
- **Mô hình AI**: YOLOv11n, mô hình nhận diện khuôn mặt tùy chỉnh, mô hình cảm xúc TFLite

## 2. Kiến trúc hệ thống

### 2.1. Sơ đồ kiến trúc

```
┌────────────────┐     ┌───────────────────┐     ┌────────────────┐
│                │     │                   │     │                │
│  Web Frontend  │◄───►│  FastAPI Backend  │◄───►│  AI Models     │
│  (HTML/JS/CSS) │     │  (Python)         │     │  (YOLO/TFLite) │
│                │     │                   │     │                │
└────────────────┘     └───────────────────┘     └────────────────┘
```

### 2.2. Luồng dữ liệu

1. Người dùng mở ứng dụng web và cho phép truy cập camera
2. Frontend chụp ảnh từ camera và gửi đến backend qua API
3. Backend sử dụng các mô hình AI để phát hiện người, khuôn mặt và cảm xúc
4. Backend trả về kết quả dưới dạng JSON
5. Frontend hiển thị kết quả lên giao diện người dùng

## 3. Các thành phần hệ thống

### 3.1. Frontend

#### 3.1.1. Cấu trúc thư mục

```
frontend/
├── index.html         # Trang chính của ứng dụng
├── css/               # Các file CSS
│   ├── style.css      # CSS chính (nhập khẩu các file CSS khác)
│   ├── base.css       # Biến và kiểu cơ bản
│   ├── layout.css     # Layout chính và responsive
│   └── components/    # CSS cho từng thành phần
│       ├── header-footer.css
│       ├── video.css
│       ├── stats.css
│       ├── controls.css
│       ├── toggles.css
│       ├── buttons.css
│       └── loading.css
└── js/                # Các file JavaScript
    ├── main.js        # Điểm khởi đầu ứng dụng
    ├── camera.js      # Xử lý camera
    ├── detection.js   # Xử lý kết quả nhận diện
    ├── stats.js       # Cập nhật thống kê
    ├── ui.js          # Xử lý giao diện người dùng
    ├── state.js       # Quản lý trạng thái ứng dụng
    └── config.js      # Cấu hình ứng dụng
```

#### 3.1.2. Các thành phần chính

- **Camera**: Quản lý việc bắt đầu, dừng và chọn camera
- **Detection**: Gửi khung hình đến backend và xử lý kết quả
- **Stats**: Hiển thị thống kê về người, khuôn mặt và cảm xúc
- **UI**: Quản lý giao diện người dùng và tương tác
- **State**: Lưu trữ trạng thái ứng dụng

### 3.2. Backend

#### 3.2.1. Cấu trúc thư mục

```
app/
├── __init__.py
├── box_detector.py    # Xử lý nhận diện người và khuôn mặt
├── config.py          # Cấu hình backend
└── models.py          # Quản lý mô hình AI

backend/
└── main.py            # Máy chủ FastAPI chính

models/
├── yolo11n.pt         # Mô hình YOLO nhận diện người
├── best_face_model.pt # Mô hình nhận diện khuôn mặt
└── emotion_model.tflite # Mô hình nhận diện cảm xúc
```

#### 3.2.2. Các thành phần chính

- **FastAPI Server**: Xử lý các yêu cầu từ frontend
- **Detector**: Phát hiện người và khuôn mặt trong khung hình
- **Emotion Recognition**: Nhận diện cảm xúc từ khuôn mặt đã phát hiện

## 4. Cài đặt và cấu hình

### 4.1. Yêu cầu hệ thống

- Python 3.10+
- Thư viện Python: FastAPI, Uvicorn, TensorFlow, OpenCV, Ultralytics (YOLO)
- Trình duyệt web hiện đại (Chrome, Firefox, Edge, Safari) có hỗ trợ WebRTC

### 4.2. Cài đặt

1. Clone mã nguồn từ kho lưu trữ:
   ```
   git clone https://github.com/hai4h/yolo-human-face-detection.git
   cd yolo-human-face-detection
   ```

2. Cài đặt các gói phụ thuộc Python(điều chỉnh theo hệ thống):
   ```
   pip install -r ultralytics tensorflow fastapi uvicorn opencv-python
   ```

3. Tải các mô hình AI vào thư mục `models/`

### 4.3. Cấu hình

Cấu hình hệ thống có thể được điều chỉnh thông qua các file sau:

- `app/config.py`: Cấu hình backend và mô hình AI
- `frontend/js/config.js`: Cấu hình frontend

## 5. Sử dụng hệ thống

### 5.1. Khởi động máy chủ

```
python run_server.py
```

Máy chủ sẽ khởi động tại địa chỉ `https://0.0.0.0:8000` với SSL được bật.

### 5.2. Truy cập ứng dụng

Mở trình duyệt và truy cập:
```
https://localhost:8000
```

### 5.3. Sử dụng ứng dụng

1. Cho phép truy cập camera khi được yêu cầu
2. Chọn camera (nếu có nhiều camera)
3. Nhấn "Bắt đầu Camera" để bắt đầu nhận diện
4. Sử dụng các tùy chọn hiển thị để điều chỉnh giao diện
5. Nhấn "Dừng Camera" để dừng nhận diện

## 6. API Backend

### 6.1. Endpoint nhận diện khung hình

- **URL**: `/process_frame`
- **Method**: POST
- **Content-Type**: `multipart/form-data`
- **Parameters**: 
  - `file`: File hình ảnh JPEG/PNG
- **Response**: JSON

```json
{
  "persons": 2,
  "faces": 1,
  "person_boxes": [
    {"coords": [x1, y1, x2, y2], "confidence": 0.95}
  ],
  "face_boxes": [
    {"coords": [x1, y1, x2, y2], "confidence": 0.92, "emotion": "Vui vẻ"}
  ]
}
```

### 6.2. Endpoint kiểm tra sức khỏe

- **URL**: `/health`
- **Method**: GET
- **Response**: JSON

```json
{
  "status": "ok"
}
```

## 7. Mô hình AI

### 7.1. Mô hình nhận diện người

- **Mô hình**: YOLOv11n
- **Định dạng**: PyTorch (.pt)
- **Lớp phát hiện**: Person (class 0)
- **Ngưỡng tin cậy**: 0.3

### 7.2. Mô hình nhận diện khuôn mặt

- **Mô hình**: YOLOv11n-Face tùy chỉnh
- **Định dạng**: PyTorch (.pt)
- **Ngưỡng tin cậy**: 0.3

### 7.3. Mô hình nhận diện cảm xúc

- **Mô hình**: CNN tùy chỉnh
- **Định dạng**: TensorFlow Lite (.tflite)
- **Kích thước đầu vào**: 32x32 pixels (grayscale)
- **Lớp cảm xúc**:
  - 0: Giận dữ
  - 1: Ghê tởm
  - 2: Sợ hãi
  - 3: Vui vẻ
  - 4: Buồn bã
  - 5: Ngạc nhiên
  - 6: Bình thường

## 8. Giao diện người dùng

### 8.1. Giao diện Desktop

Giao diện desktop được chia thành ba cột:
- **Cột trái**: Thống kê (số người, khuôn mặt, cảm xúc)
- **Cột giữa**: Khung video từ camera
- **Cột phải**: Bảng điều khiển với các tùy chọn

### 8.2. Giao diện Mobile

Giao diện mobile được chia thành các phần theo thứ tự từ trên xuống dưới:
- **Thống kê**: Hiển thị số lượng và cảm xúc
- **Video**: Khung hình từ camera
- **Điều khiển**: Các tùy chọn và nút điều khiển

## 9. Xử lý lỗi và gỡ rối

### 9.1. Lỗi phổ biến

- **Không tìm thấy camera**: Kiểm tra quyền truy cập camera trong trình duyệt
- **Mô hình không tải được**: Kiểm tra đường dẫn mô hình trong `config.py`
- **Độ trễ cao**: Giảm kích thước khung hình hoặc tốc độ khung hình trong `config.js`

### 9.2. Xem nhật ký

- Nhật ký backend: Terminal nơi chạy `run_server.py`
- Nhật ký frontend: Console trong DevTools của trình duyệt

## 10. Tối ưu hóa hiệu suất

### 10.1. Tối ưu hóa frontend

- Giảm `frameRate` trong `config.js` trên thiết bị có hiệu suất thấp
- Điều chỉnh kích thước khung video để giảm dữ liệu truyền tải

### 10.2. Tối ưu hóa backend

- Sử dụng mô hình TensorFlow Lite thay vì mô hình TensorFlow đầy đủ
- Bật `half=True` khi sử dụng mô hình YOLO để tính toán ở độ chính xác FP16
- Giảm kích thước ảnh khi xử lý

## 11. Bảo mật

- Hệ thống sử dụng SSL (HTTPS) để mã hóa dữ liệu
- Dữ liệu video không được lưu trữ
- Tất cả xử lý đều diễn ra cục bộ trên máy chủ