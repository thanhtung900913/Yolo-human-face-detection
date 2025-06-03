from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
putenv("ROCM_PATH", "/opt/rocm-6.3.3")

import numpy as np
import tensorflow as tf
import time
import os
import glob
from pathlib import Path
import argparse

def benchmark_tflite_model(model_path, num_runs=100, use_real_data=True, data_path=None):
    # Load mô hình TFLite
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Lấy thông tin về input và output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # In thông tin input và output
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
    
    # Lấy kích thước input
    if input_details[0]['shape_signature'][1] is None:
        # Nếu mô hình có kích thước linh hoạt, chúng ta chọn kích thước phổ biến
        input_height, input_width = 224, 224
    else:
        input_height = input_details[0]['shape_signature'][1]
        input_width = input_details[0]['shape_signature'][2]
    
    # Lưu ý số kênh màu từ mô hình (grayscale = 1, RGB = 3)
    input_channels = input_details[0]['shape_signature'][3]
    input_shape = (1, input_height, input_width, input_channels)
    print(f"Using input shape: {input_shape}")
    
    # Kiểm tra định dạng quantization
    is_quantized = input_details[0]['dtype'] == np.int8 or input_details[0]['dtype'] == np.uint8
    
    # Tạo dữ liệu đầu vào
    if use_real_data and data_path:
        # Tìm tất cả các file ảnh .jpg
        image_files = glob.glob(os.path.join(data_path, "**/*.jpg"), recursive=True)
        if image_files:
            print(f"Found {len(image_files)} real images, using the first one for benchmark")
            img_path = image_files[0]
            
            # Đọc ảnh
            img = tf.io.read_file(img_path)
            
            # Xử lý ảnh dựa vào số kênh màu
            if input_channels == 1:
                # Đọc dưới dạng grayscale
                img = tf.image.decode_jpeg(img, channels=1)
            else:
                # Đọc dưới dạng RGB
                img = tf.image.decode_jpeg(img, channels=3)
                
            img = tf.image.resize(img, [input_height, input_width])
            img = tf.cast(img, tf.float32) / 255.0
            input_data = img.numpy().reshape(input_shape)
            
            # Quantize input nếu cần
            if is_quantized:
                # Lấy tham số zero point và scale
                scale, zero_point = input_details[0]['quantization']
                input_data = input_data / scale + zero_point
                input_data = input_data.astype(input_details[0]['dtype'])
        else:
            print("No images found, using random data")
            use_real_data = False
    
    if not use_real_data:
        # Tạo dữ liệu ngẫu nhiên
        if is_quantized:
            # Sinh dữ liệu Int8
            input_data = np.random.randint(-128, 127, input_shape).astype(input_details[0]['dtype'])
        else:
            # Sinh dữ liệu Float32
            input_data = np.random.random(input_shape).astype(np.float32)
    
    # Warm up
    print("Warming up...")
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
    
    # Benchmark
    print(f"Running benchmark with {num_runs} iterations...")
    start_time = time.time()
    
    for i in range(num_runs):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
    
    end_time = time.time()
    
    # Tính thời gian
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    fps = 1.0 / avg_time
    
    # Lấy kết quả từ lần chạy cuối
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # In kết quả
    print(f"\nBenchmark Results for {Path(model_path).name}:")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    
    # In thêm thông tin về model
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Model type: {'Quantized (INT8)' if is_quantized else 'Float (FP32)'}")
    
    # In kết quả inference
    if output_data.size <= 20:  # Chỉ in nếu output không quá lớn
        if is_quantized:
            scale, zero_point = output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - zero_point) * scale
        
        print("\nInference result:")
        print(output_data)
        
        # Nếu đây là mô hình phân loại, in class có xác suất cao nhất
        if output_data.shape[-1] > 1:
            predicted_class = np.argmax(output_data)
            confidence = output_data.flatten()[predicted_class]
            print(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")
    
    return avg_time, fps, model_size_mb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark TFLite model')
    parser.add_argument('--model', type=str, default='models/human_action_recognition_model.tflite',
                        help='Path to TFLite model')
    parser.add_argument('--runs', type=int, default=100,
                        help='Number of runs for benchmark')
    parser.add_argument('--use_real_data', action='store_true', default=True,
                        help='Use real image data if available')
    parser.add_argument('--data_path', type=str, 
                        default='/home/haiah/Workspace/yolo-human-face-detection/dataset/har/test',
                        help='Path to test images')
    
    args = parser.parse_args()
    
    benchmark_tflite_model(args.model, args.runs, args.use_real_data, args.data_path)