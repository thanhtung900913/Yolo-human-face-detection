# Đường dẫn mô hình / Model paths
PERSON_MODEL_PATH = './models/yolov8n-person-lw.pt'
FACE_MODEL_PATH = './models/yolov8p-face-v2.pt'
EMOTION_MODEL_PATH = './models/emotion_model_2.tflite'
ACTION_MODEL_PATH = './models/human_action_recognition_model.tflite'
EMBEDDING_MODEL_PATH = './models/face_embedding_model.tflite'

# Thiết lập hiển thị / Visualization settings
PERSON_COLOR = (0, 0, 255)   # Màu đỏ / Red color
FACE_COLOR = (0, 255, 0)     # Màu xanh lá / Green color
EMOTION_COLOR = (255, 165, 0)  # Màu cam / Orange color for emotion
TEXT_SCALE = 1             # Tỷ lệ chữ / Text scale
TEXT_THICKNESS = 2           # Độ dày chữ / Text thickness

# Danh sách cảm xúc / Emotion labels
EMOTION_LABELS = ['Giận dữ', 'Ghê tởm', 'Sợ hãi', 'Vui vẻ', 'Buồn bã', 'Ngạc nhiên', 'Bình thường']

# Danh sách hành vi / Action labels
ACTION_LABELS = ['Gọi điện', 'Vỗ tay', 'Đạp xe', 'Khiêu vũ', 'Uống nước', 
                'Ăn uống', 'Đánh nhau', 'Ôm', 'Cười', 'Nghe nhạc', 
                'Chạy', 'Ngồi', 'Ngủ', 'Nhắn tin', 'Dùng laptop']
REDIS_CONFIG = {
    "host": "localhost", 
    "port": 6379,
    "db": 0,
    "password": None,  # Đặt mật khẩu nếu có / Set password if needed
    "decode_responses": False,  # Không tự động giải mã kết quả / Don't auto-decode responses
}

# Cấu hình Vector Search / Vector Search configuration
VECTOR_CONFIG = {
    "default_index": "Giang_128",
    "face_vector_dim": 128,
    "default_top_k": 5,
    "min_similarity_score": 0.7,  # Ngưỡng điểm tương đồng tối thiểu / Minimum similarity score threshold
}

# Cấu hình Redis Stream / Redis Stream configuration  
STREAM_CONFIG = {
    "face_stream": "face_events",
    "log_stream": "system_logs",
    "max_stream_length": 1000,  # Độ dài tối đa lưu trữ / Maximum stream length
}