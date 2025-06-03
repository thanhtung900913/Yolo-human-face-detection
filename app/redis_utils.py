import redis
import time
import json
import numpy as np
from datetime import datetime
from .config import REDIS_CONFIG, VECTOR_CONFIG, STREAM_CONFIG

class RedisManager:
    """Quản lý kết nối và truy vấn Redis"""
    
    def __init__(self, host=None, port=None, password=None, db=None):
        """
        Khởi tạo kết nối với Redis
        
        Args:
            host (str, optional): Địa chỉ Redis host
            port (int, optional): Cổng Redis
            password (str, optional): Mật khẩu Redis
            db (int, optional): Chỉ số Redis db
        """
        # Sử dụng giá trị từ tham số nếu có, ngược lại sử dụng từ config
        self.redis_config = {
            "host": host or REDIS_CONFIG["host"],
            "port": port or REDIS_CONFIG["port"],
            "db": db if db is not None else REDIS_CONFIG["db"],
            "password": password or REDIS_CONFIG["password"],
            "decode_responses": REDIS_CONFIG["decode_responses"]
        }
        
        # Xóa các giá trị None để tránh lỗi khi kết nối
        self.redis_config = {k: v for k, v in self.redis_config.items() if v is not None}
        
        # Tạo kết nối Redis
        self._connect()

    def _connect(self):
        """Tạo kết nối tới Redis server"""
        try:
            self.redis_client = redis.Redis(**self.redis_config)
            self.connected = self.redis_client.ping()
            if self.connected:
                print(f"Kết nối Redis thành công: {self.redis_config['host']}:{self.redis_config['port']}")
            else:
                print("Kết nối Redis thất bại: không thể ping tới server")
        except Exception as e:
            self.connected = False
            self.redis_client = None
            print(f"Lỗi kết nối Redis: {e}")

    def is_connected(self):
        """Kiểm tra trạng thái kết nối với Redis"""
        if not self.redis_client:
            return False
        
        try:
            return self.redis_client.ping()
        except:
            self.connected = False
            return False

    def reconnect(self):
        """Thử kết nối lại Redis nếu mất kết nối"""
        if not self.is_connected():
            self._connect()
        return self.connected

    def search_vector(self, vector, index_name=None, top_k=None):
        """
        Tìm kiếm khuôn mặt tương tự trong Redis Vector Search
        
        Args:
            vector (np.ndarray): Vector đặc trưng cần tìm kiếm
            index_name (str, optional): Tên chỉ mục Redis Search
            top_k (int, optional): Số lượng kết quả trả về
            
        Returns:
            dict: Kết quả tìm kiếm
        """
        # Kiểm tra kết nối
        if not self.is_connected():
            if not self.reconnect():
                return {
                    "success": False,
                    "message": "Không thể kết nối tới Redis",
                    "results": [],
                    "query_time_ms": 0
                }
        
        try:
            # Sử dụng giá trị mặc định từ config nếu không có tham số
            index_name = index_name or VECTOR_CONFIG["default_index"]
            top_k = top_k or VECTOR_CONFIG["default_top_k"]
            
            # Đảm bảo vector là một mảng numpy kiểu float32
            if not isinstance(vector, np.ndarray):
                vector = np.array(vector, dtype=np.float32)
            elif vector.dtype != np.float32:
                vector = vector.astype(np.float32)
                
            # Đo thời gian truy vấn
            start_time = time.time()
            
            # Thực hiện truy vấn KNN
            res = self.redis_client.execute_command(
                "FT.SEARCH", index_name,
                f"*=>[KNN {top_k} @embedding $vec_param]",
                "PARAMS", 2, "vec_param", vector.tobytes(),
                "DIALECT", 2,
                "SORTBY", "__embedding_score",
                "RETURN", 3, "timestamp", "name", "__embedding_score"
            )
            
            # Tính thời gian truy vấn
            elapsed_time_ms = (time.time() - start_time) * 1000
            
            # Xử lý kết quả
            results = []
            
            if len(res) > 1:
                for i in range(1, len(res), 2):
                    try:
                        doc_id = res[i].decode()
                        fields_raw = res[i + 1]
                        
                        # Chuyển đổi danh sách thành dictionary
                        fields = {}
                        for j in range(0, len(fields_raw), 2):
                            if j+1 < len(fields_raw):
                                key = fields_raw[j]
                                value = fields_raw[j + 1]
                                
                                if isinstance(key, bytes):
                                    key = key.decode()
                                    
                                if isinstance(value, bytes):
                                    value = value.decode()
                                    
                                fields[key] = value
                        
                        # Lấy các trường dữ liệu
                        name = fields.get('name', 'N/A')
                        timestamp = fields.get('timestamp', 'N/A')
                        score = float(fields.get('__embedding_score', '0'))
                        
                        # Thêm vào danh sách kết quả
                        results.append({
                            "id": doc_id,
                            "name": name,
                            "timestamp": timestamp,
                            "score": score,
                        })
                    except Exception as e:
                        print(f"Lỗi khi xử lý kết quả: {e}")
                        continue
            
            return {
                "success": True,
                "message": "Tìm kiếm thành công" if results else "Không tìm thấy kết quả phù hợp",
                "results": results,
                "query_time_ms": elapsed_time_ms
            }
            
        except Exception as e:
            print(f"Lỗi khi tìm kiếm vector: {e}")
            return {
                "success": False,
                "message": f"Lỗi khi tìm kiếm: {str(e)}",
                "results": [],
                "query_time_ms": 0
            }
    
    