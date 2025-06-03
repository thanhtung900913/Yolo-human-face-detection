import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import redis
import numpy as np
import cv2
import uuid
from datetime import datetime
from app.box_detector import Detector

# Tạo một instance của Detector
detector = Detector()

# Kết nối Redis
r = redis.Redis(host='localhost', port=6379)

# Tên index bạn muốn tạo
index_name = "test_index4"  

# Kiểm tra nếu index chưa tồn tại thì mới tạo
if index_name.encode() not in r.execute_command("FT._LIST"):
    try:
        r.execute_command(
            "FT.CREATE", index_name,
            "ON", "HASH",
            "PREFIX", "1", "person:", 
            "SCHEMA",
            "embedding", "VECTOR", "HNSW", "6",
            "TYPE", "FLOAT32", "DIM", "128", "DISTANCE_METRIC", "COSINE",
            "timestamp", "TEXT",
            "name", "TEXT"
        )
        print(f"Đã tạo index {index_name}")
    except redis.exceptions.ResponseError as e:
        print("Lỗi khi tạo index:", e)

else:
    print(f"Index '{index_name}' đã tồn tại.")

def upload_face_vector(image_path, person_name):
    try:
        if not os.path.exists(image_path):
            return {"success": False, "message": f"Không tìm thấy ảnh: {image_path}"}
        
        image = cv2.imread(image_path)
        if image is None:
            return {"success": False, "message": f"Không thể đọc ảnh: {image_path}"}
        
        face_results = detector.face_model(image, conf=0.3, iou=0.45, imgsz=160, half=True, verbose=False)
        if not face_results or len(face_results[0].boxes) == 0:
            return {"success": False, "message": "Không tìm thấy khuôn mặt trong ảnh"}
        
        face_box = face_results[0].boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, face_box)
        face_roi = image[y1:y2, x1:x2]
        
        face_vector = detector.extract_face_vector(face_roi)
        
        counter_key = f"counter:{person_name}"
        current_index = r.incr(counter_key)
        
        # Key trong Redis sẽ là "person:PERSON_NAME_INDEX"
        redis_key = f"person:{person_name}_{current_index}"
        
        vector_blob = face_vector.tobytes()
        timestamp = datetime.now().isoformat()
        
        r.hset(
            redis_key,
            mapping={
                "embedding": vector_blob,
                "timestamp": timestamp,
                "name": person_name  # Lưu tên người thật vào field name
            }
        )
        
        print(f"Đã upload vector khuôn mặt cho '{person_name}' với ID: {redis_key}")
        return {
            "success": True,
            "message": f"Đã upload thành công vector khuôn mặt cho '{person_name}'",
            "vector_id": redis_key
        }

    except Exception as e:
        return {"success": False, "message": f"Lỗi khi upload vector khuôn mặt: {str(e)}"}
        
        print(f"Đã upload vector khuôn mặt cho '{person_name}' với ID: {vector_id}")
        return {
            "success": True,
            "message": f"Đã upload thành công vector khuôn mặt cho '{person_name}'",
            "vector_id": vector_id
        }

    except Exception as e:
        return {"success": False, "message": f"Lỗi khi upload vector khuôn mặt: {str(e)}"}

# HÀM MAIN
def main():
    data_root = r"C:\Users\Admin\Downloads\drive-download-20250519T131600Z-1-001"

    for person_name in os.listdir(data_root):
        person_folder = os.path.join(data_root, person_name)

        if not os.path.isdir(person_folder):
            continue

        for filename in os.listdir(person_folder):
            if filename.lower().endswith((".jpg", ".png")):
                image_path = os.path.join(person_folder, filename)
                result = upload_face_vector(image_path, person_name)

                if result["success"]:
                    print(result["message"])
                    print(f"Vector ID: {result['vector_id']}")
                else:
                    print(f"Lỗi ({person_name} - {filename}): {result['message']}")


if __name__ == "__main__":
    main()