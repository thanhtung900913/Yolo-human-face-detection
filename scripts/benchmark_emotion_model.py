"""
Benchmark script for emotion recognition model latency testing
Script đo hiệu năng độ trễ (latency) cho mô hình nhận diện cảm xúc
Usage/Cách sử dụng:
    python benchmark_emotion_model.py --model_path path/to/model.[h5|tflite] --data_path path/to/test_images
    e.g. python3 ./scripts/benchmark_emotion_model.py --model_path model.h5 --data_path ./dataset/fer-2013/test
"""
import os
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from glob import glob

# Cấu hình môi trường cho AMD GPU / Environment setup for AMD GPUs
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["ROCM_PATH"] = "/opt/rocm-6.3.3"

def preprocess_image(image_path, target_size=(32, 32)):
    """
    Preprocess a single image for emotion prediction
    Tiền xử lý một ảnh để dự đoán cảm xúc
    
    Args:
        image_path (str): Path to the image file / Đường dẫn đến file ảnh
        target_size (tuple): Target size for resizing / Kích thước mục tiêu để resize
        
    Returns:
        numpy.ndarray: Preprocessed image / Ảnh đã được tiền xử lý
    """
    # Load and resize image / Tải và resize ảnh
    img = load_img(image_path, color_mode='grayscale', target_size=target_size)
    
    # Convert to array and normalize / Chuyển đổi thành mảng và chuẩn hóa
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1] / Chuẩn hóa về [0, 1]
    
    # Add batch dimension / Thêm chiều batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def load_tflite_model(model_path):
    """
    Load TensorFlow Lite model
    Tải mô hình TensorFlow Lite
    
    Args:
        model_path (str): Path to the .tflite model file / Đường dẫn đến file mô hình .tflite
        
    Returns:
        interpreter: TFLite interpreter / Trình thông dịch TFLite
    """
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    return interpreter

def get_model_type(model_path):
    """
    Determine the model type based on file extension
    Xác định loại mô hình dựa trên phần mở rộng file
    
    Args:
        model_path (str): Path to the model file / Đường dẫn đến file mô hình
        
    Returns:
        str: Model type ('h5' or 'tflite') / Loại mô hình ('h5' hoặc 'tflite')
    """
    _, ext = os.path.splitext(model_path)
    if ext.lower() == '.tflite':
        return 'tflite'
    elif ext.lower() == '.h5':
        return 'h5'
    else:
        raise ValueError(f"Unsupported model format: {ext}. Supported formats are .h5 and .tflite")

def predict_with_tflite(interpreter, input_data):
    """
    Run inference with TensorFlow Lite model
    Chạy dự đoán với mô hình TensorFlow Lite
    
    Args:
        interpreter: TFLite interpreter / Trình thông dịch TFLite
        input_data (numpy.ndarray): Input image data / Dữ liệu ảnh đầu vào
        
    Returns:
        numpy.ndarray: Model prediction / Dự đoán của mô hình
    """
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Check if the input shape is compatible
    input_shape = input_details[0]['shape']
    if input_data.shape != (1, input_shape[1], input_shape[2], input_shape[3]):
        input_data = tf.image.resize(input_data, (input_shape[1], input_shape[2]))
        input_data = tf.reshape(input_data, (1, input_shape[1], input_shape[2], input_shape[3]))
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return output_data

def benchmark_model(model, image_paths, num_runs=100, model_type='h5'):
    """
    Benchmark model prediction latency
    Đo hiệu năng độ trễ dự đoán của mô hình
    
    Args:
        model: Loaded model (TensorFlow model or TFLite interpreter) / Mô hình đã tải
        image_paths (list): List of image paths / Danh sách các đường dẫn ảnh
        num_runs (int): Number of runs for the benchmark / Số lần chạy cho benchmark
        model_type (str): Type of model ('h5' or 'tflite') / Loại mô hình ('h5' hoặc 'tflite')
        
    Returns:
        dict: Dictionary containing benchmark results / Dictionary chứa kết quả benchmark
    """
    # Pre-load some random images to test / Tải trước một số ảnh ngẫu nhiên để kiểm tra
    test_images = np.random.choice(image_paths, min(num_runs, len(image_paths)))
    
    # Warm up the model / Khởi động mô hình
    img = preprocess_image(test_images[0])
    if model_type == 'h5':
        for _ in range(10):
            _ = model.predict(img, verbose=0)
    else:  # tflite
        for _ in range(10):
            _ = predict_with_tflite(model, img)
    
    # Lists to store timing results / Danh sách để lưu kết quả thời gian
    preprocessing_times = []
    prediction_times = []
    total_times = []
    
    # Run benchmark / Chạy benchmark
    print(f"Bắt đầu benchmark với {num_runs} lần chạy...")
    for i in range(num_runs):
        # Choose an image / Chọn một ảnh
        img_path = test_images[i % len(test_images)]
        
        # Measure preprocessing time / Đo thời gian tiền xử lý
        preprocess_start = time.time()
        img = preprocess_image(img_path)
        preprocess_end = time.time()
        preprocess_time = (preprocess_end - preprocess_start) * 1000  # Convert to ms / Chuyển đổi sang ms
        
        # Measure prediction time / Đo thời gian dự đoán
        predict_start = time.time()
        if model_type == 'h5':
            _ = model.predict(img, verbose=0)
        else:  # tflite
            _ = predict_with_tflite(model, img)
        predict_end = time.time()
        predict_time = (predict_end - predict_start) * 1000  # Convert to ms / Chuyển đổi sang ms
        
        # Calculate total time / Tính tổng thời gian
        total_time = preprocess_time + predict_time
        
        # Store times / Lưu các thời gian
        preprocessing_times.append(preprocess_time)
        prediction_times.append(predict_time)
        total_times.append(total_time)
        
        # Print progress / In tiến độ
        if (i + 1) % 10 == 0:
            print(f"Đã chạy: {i + 1}/{num_runs}")
    
    # Calculate statistics / Tính toán thống kê
    avg_preprocess = np.mean(preprocessing_times)
    avg_predict = np.mean(prediction_times)
    avg_total = np.mean(total_times)
    
    median_predict = np.median(prediction_times)
    p95_predict = np.percentile(prediction_times, 95)
    p99_predict = np.percentile(prediction_times, 99)
    
    # Calculate estimated FPS / Tính FPS ước tính
    fps_predict = 1000 / avg_predict
    fps_total = 1000 / avg_total
    
    # Create results dictionary / Tạo dictionary kết quả
    results = {
        "avg_preprocess": avg_preprocess,
        "avg_predict": avg_predict,
        "avg_total": avg_total,
        "median_predict": median_predict,
        "p95_predict": p95_predict,
        "p99_predict": p99_predict,
        "fps_predict": fps_predict,
        "fps_total": fps_total,
        "prediction_times": prediction_times,
        "model_type": model_type
    }
    
    return results

def display_results(results):
    """
    Display benchmark results
    Hiển thị kết quả benchmark
    
    Args:
        results (dict): Dictionary containing benchmark results / Dictionary chứa kết quả benchmark
    """
    model_type_str = "TensorFlow (.h5)" if results['model_type'] == 'h5' else "TensorFlow Lite (.tflite)"
    
    print("\n===== KẾT QUẢ BENCHMARK =====")
    print(f"Loại mô hình: {model_type_str}")
    print(f"Số lần chạy: 100")
    print(f"Thời gian tiền xử lý trung bình: {results['avg_preprocess']:.2f} ms")
    print(f"Thời gian dự đoán trung bình: {results['avg_predict']:.2f} ms")
    print(f"Tổng thời gian xử lý trung bình: {results['avg_total']:.2f} ms")
    print(f"Median (P50) thời gian dự đoán: {results['median_predict']:.2f} ms")
    print(f"P95 thời gian dự đoán: {results['p95_predict']:.2f} ms")
    print(f"P99 thời gian dự đoán: {results['p99_predict']:.2f} ms")
    print(f"FPS ước tính (chỉ tính dự đoán): {results['fps_predict']:.2f}")
    print(f"FPS ước tính (bao gồm tiền xử lý): {results['fps_total']:.2f}")
    print("==============================")
    
    # Provide recommendations based on latency / Đưa ra các khuyến nghị dựa trên độ trễ
    if results['fps_total'] < 30:
        print("CẢNH BÁO: Độ trễ hiện tại có thể không đủ cho ứng dụng thời gian thực ở 30 FPS.")
        print("Bạn có thể cân nhắc các phương pháp tối ưu hoá sau:")
        if results['model_type'] == 'h5':
            print("1. Chuyển đổi mô hình sang TensorFlow Lite (.tflite)")
        print(f"2. Sử dụng kích thước đầu vào nhỏ hơn (24x24 thay vì 32x32)")
        print(f"3. Lượng tử hoá mô hình (quantization)")
        print(f"4. Distillation sang mô hình nhẹ hơn")
        if results['model_type'] == 'h5':
            print(f"5. Sử dụng TensorRT (NVIDIA GPU) hoặc ONNX Runtime để tăng tốc")
    else:
        print(f"Mô hình đáp ứng được yêu cầu xử lý thời gian thực ở 30 FPS.")

def plot_latency_distribution(prediction_times, model_type):
    """
    Plot the distribution of prediction latencies
    Vẽ biểu đồ phân phối độ trễ dự đoán
    
    Args:
        prediction_times (list): List of prediction times / Danh sách các thời gian dự đoán
        model_type (str): Type of model ('h5' or 'tflite') / Loại mô hình ('h5' hoặc 'tflite')
    """
    model_type_str = "TensorFlow" if model_type == 'h5' else "TensorFlow Lite"
    filename = f"latency_distribution_{model_type}.png"
    
    plt.figure(figsize=(10, 6))
    plt.hist(prediction_times, bins=20, alpha=0.7, color='blue')
    plt.axvline(np.mean(prediction_times), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(prediction_times):.2f} ms')
    plt.axvline(np.median(prediction_times), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(prediction_times):.2f} ms')
    plt.axvline(np.percentile(prediction_times, 95), color='orange', linestyle='dashed', linewidth=2, label=f'P95: {np.percentile(prediction_times, 95):.2f} ms')
    
    plt.title(f'Phân phối độ trễ dự đoán - {model_type_str} (Prediction Latency Distribution)')
    plt.xlabel('Thời gian dự đoán (ms)')
    plt.ylabel('Số lượng')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot / Lưu biểu đồ
    plt.savefig(filename)
    plt.close()
    
    print(f"Đã lưu biểu đồ phân phối độ trễ vào '{filename}'")

def main():
    """Main function / Hàm chính"""
    parser = argparse.ArgumentParser(description='Benchmark emotion recognition model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to emotion model (.h5 or .tflite)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test images folder')
    parser.add_argument('--num_runs', type=int, default=100, help='Number of runs for benchmark')
    parser.add_argument('--input_size', type=int, default=32, help='Input image size (default: 32)')
    
    args = parser.parse_args()
    
    # Check if model exists / Kiểm tra xem mô hình có tồn tại không
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    # Check if data path exists / Kiểm tra xem đường dẫn dữ liệu có tồn tại không
    if not os.path.exists(args.data_path):
        print(f"Error: Data directory not found at {args.data_path}")
        return
    
    # Detect model type / Xác định loại mô hình
    try:
        model_type = get_model_type(args.model_path)
        print(f"Detected model type: {model_type}")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Find image files / Tìm các file ảnh
    image_files = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_files.extend(glob(os.path.join(args.data_path, f'**/*.{ext}'), recursive=True))
    
    if not image_files:
        print(f"Error: No image files found in {args.data_path}")
        return
    
    print(f"Tìm thấy {len(image_files)} ảnh. Bắt đầu benchmark...")
    
    # Load model based on type / Tải mô hình dựa trên loại
    try:
        if model_type == 'h5':
            model = load_model(args.model_path)
            
            # Optimize model for inference / Tối ưu hóa mô hình cho inference
            inference_model = tf.keras.models.clone_model(model)
            inference_model.set_weights(model.get_weights())
            
            # Run benchmark / Chạy benchmark
            results = benchmark_model(
                inference_model, 
                image_files, 
                num_runs=args.num_runs,
                model_type=model_type
            )
        else:  # tflite
            interpreter = load_tflite_model(args.model_path)
            
            # Run benchmark / Chạy benchmark
            results = benchmark_model(
                interpreter, 
                image_files, 
                num_runs=args.num_runs,
                model_type=model_type
            )
        
        # Display results / Hiển thị kết quả
        display_results(results)
        
        # Plot latency distribution / Vẽ biểu đồ phân phối độ trễ
        plot_latency_distribution(results['prediction_times'], model_type)
        
    except Exception as e:
        print(f"Error during benchmark: {e}")

if __name__ == "__main__":
    main()