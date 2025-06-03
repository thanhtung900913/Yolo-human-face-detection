from os import putenv

# Cấu hình môi trường cho AMD GPU / Environment setup for AMD GPUs
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
putenv("ROCM_PATH", "/opt/rocm-6.3.0/")

import tensorflow as tf
import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def load_and_run_model(model_path, thread_id):
    print(f"Thread {thread_id} bắt đầu...")
    start_time = time.time()
    
    # Tạo interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Lấy thông tin đầu vào/ra
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Lấy shape đầu vào từ model
    input_shape = input_details[0]['shape']
    print(f"Thread {thread_id} - Input shape: {input_shape}")
    
    # Tạo dữ liệu đầu vào phù hợp với kích thước yêu cầu
    input_data = np.random.random(input_shape).astype(np.float32)
    
    # Đặt input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Chạy inference
    print(f"Thread {thread_id} bắt đầu inference...")
    interpreter.invoke()
    
    # Lấy kết quả
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"Thread {thread_id} hoàn thành trong {time.time() - start_time:.4f} giây")
    print(f"Thread {thread_id} - Output shape: {output_data.shape}")
    return output_data, interpreter, input_details

# Đường dẫn đến model
emotion_model_path = "./models/emotion_model_2.tflite"
action_model_path = "./models/human_action_recognition_model.tflite"

print("=== WARM UP PHASE ===")
# Tạo và khởi tạo các interpreter để làm nóng
emotion_interpreter = tf.lite.Interpreter(model_path=emotion_model_path)
emotion_interpreter.allocate_tensors()
emotion_input = emotion_interpreter.get_input_details()
emotion_shape = emotion_input[0]['shape']
emotion_data = np.random.random(emotion_shape).astype(np.float32)
emotion_interpreter.set_tensor(emotion_input[0]['index'], emotion_data)
print("Làm nóng model emotion...")
emotion_interpreter.invoke()
print("Đã làm nóng model emotion")

action_interpreter = tf.lite.Interpreter(model_path=action_model_path)
action_interpreter.allocate_tensors()
action_input = action_interpreter.get_input_details()
action_shape = action_input[0]['shape']
action_data = np.random.random(action_shape).astype(np.float32)
action_interpreter.set_tensor(action_input[0]['index'], action_data)
print("Làm nóng model action...")
action_interpreter.invoke()
print("Đã làm nóng model action")

# Đợi một chút để đảm bảo các tài nguyên được giải phóng
time.sleep(1)
print("Đã hoàn thành warm-up\n")

# Test 1: Chạy tuần tự
print("=== CHẠY TUẦN TỰ (SAU KHI WARM-UP) ===")
start = time.time()
load_and_run_model(emotion_model_path, "Emotion")
load_and_run_model(action_model_path, "Action")
print(f"Tổng thời gian chạy tuần tự: {time.time() - start:.4f} giây")

# Test 2: Chạy song song
print("\n=== CHẠY SONG SONG (SAU KHI WARM-UP) ===")
start = time.time()
with ThreadPoolExecutor(max_workers=2) as executor:
    future1 = executor.submit(load_and_run_model, emotion_model_path, "Emotion")
    future2 = executor.submit(load_and_run_model, action_model_path, "Action")
    
    # Lấy kết quả (để đảm bảo cả hai đã hoàn thành)
    result1 = future1.result()
    result2 = future2.result()
print(f"Tổng thời gian chạy song song: {time.time() - start:.4f} giây")

# Để đánh giá tốt hơn, chúng ta sẽ chạy nhiều lần
print("\n=== BENCHMARK NHIỀU LẦN (SAU KHI WARM-UP) ===")

def run_inference(interpreter, input_data, input_index, num_runs=100):
    times = []
    for _ in range(num_runs):
        interpreter.set_tensor(input_index, input_data)
        start = time.time()
        interpreter.invoke()
        times.append(time.time() - start)
    return np.mean(times), np.min(times), np.max(times)

def run_parallel_inference(emotion_interp, emotion_data, emotion_idx, 
                        action_interp, action_data, action_idx, num_runs=100):
    times = []
    for _ in range(num_runs):
        start = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(emotion_interp.invoke)
            future2 = executor.submit(action_interp.invoke)
            future1.result()
            future2.result()
        times.append(time.time() - start)
    return np.mean(times), np.min(times), np.max(times)

# Tạo lại interpreters
emotion_output, emotion_interp, emotion_input_details = load_and_run_model(emotion_model_path, "Emotion-Benchmark")
action_output, action_interp, action_input_details = load_and_run_model(action_model_path, "Action-Benchmark")

# Chuẩn bị dữ liệu đầu vào
emotion_input_data = np.random.random(emotion_input_details[0]['shape']).astype(np.float32)
action_input_data = np.random.random(action_input_details[0]['shape']).astype(np.float32)

# Benchmark riêng lẻ
print("\nEmotion model (100 lần):")
emotion_mean, emotion_min, emotion_max = run_inference(
    emotion_interp, emotion_input_data, emotion_input_details[0]['index'])
print(f"  Thời gian trung bình: {emotion_mean*1000:.4f}ms")
print(f"  Thời gian tối thiểu: {emotion_min*1000:.4f}ms")
print(f"  Thời gian tối đa: {emotion_max*1000:.4f}ms")

print("\nAction model (100 lần):")
action_mean, action_min, action_max = run_inference(
    action_interp, action_input_data, action_input_details[0]['index'])
print(f"  Thời gian trung bình: {action_mean*1000:.4f}ms")
print(f"  Thời gian tối thiểu: {action_min*1000:.4f}ms")
print(f"  Thời gian tối đa: {action_max*1000:.4f}ms")

# Benchmark song song
print("\nSong song (100 lần):")
parallel_mean, parallel_min, parallel_max = run_parallel_inference(
    emotion_interp, emotion_input_data, emotion_input_details[0]['index'],
    action_interp, action_input_data, action_input_details[0]['index'])
print(f"  Thời gian trung bình: {parallel_mean*1000:.4f}ms")
print(f"  Thời gian tối thiểu: {parallel_min*1000:.4f}ms")
print(f"  Thời gian tối đa: {parallel_max*1000:.4f}ms")

# So sánh hiệu suất
sequential_time = emotion_mean + action_mean
speedup = sequential_time / parallel_mean
print(f"\nTổng thời gian tuần tự (emotion + action): {sequential_time*1000:.4f}ms")
print(f"Thời gian song song: {parallel_mean*1000:.4f}ms")
print(f"Tăng tốc: {speedup:.4f}x")