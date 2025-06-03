"""
Script huấn luyện mô hình nhận diện cảm xúc khuôn mặt sử dụng TensorFlow
Facial Emotion Recognition model training script using TensorFlow

Tập dữ liệu: FER2013 dataset (https://www.kaggle.com/datasets/msambare/fer2013)
Dataset: FER2013 dataset (https://www.kaggle.com/datasets/msambare/fer2013)

Cách sử dụng / Usage:
    python train_emotion_model.py --data_path /path/to/fer2013 --epochs 50 --batch_size 64 --output model.h5
"""

from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
putenv("ROCM_PATH", "/opt/rocm-6.3.3")

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization,
    DepthwiseConv2D, GlobalAveragePooling2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

def create_model(input_shape=(48, 48, 1)):
    """
    Tạo mô hình CNN cho nhận diện cảm xúc - tối ưu cho hiệu suất thấp.
    Creates a CNN model for emotion recognition optimized for low latency.
    
    Args:
        input_shape (tuple): Kích thước đầu vào của mô hình. Input shape for the model.
        
    Returns:
        model: Mô hình TensorFlow đã được biên dịch. Compiled TensorFlow model.
    """
    model = Sequential()
    
    # Block 1 - Depthwise Separable
    model.add(DepthwiseConv2D(kernel_size=(5, 5), padding='same', input_shape=input_shape))
    model.add(Conv2D(16, kernel_size=(1, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    
    # Block 2 - Depthwise Separable
    model.add(DepthwiseConv2D(kernel_size=(3, 3), padding='same'))
    model.add(Conv2D(32, kernel_size=(1, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    
    # Block 3 - Depthwise Separable
    model.add(DepthwiseConv2D(kernel_size=(3, 3), padding='same'))
    model.add(Conv2D(64, kernel_size=(1, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    # Output Block
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(7, activation='softmax'))  # 7 emotion classes
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_data(data_path, validation_split=0.2):
    """
    Tải dữ liệu FER2013.
    Loads FER2013 dataset with train and test directories.
    
    Args:
        data_path (str): Đường dẫn đến thư mục dữ liệu. Path to data directory.
        validation_split (float): Tỷ lệ dữ liệu dùng cho validation từ tập train.
        
    Returns:
        tuple: train_generator, validation_generator, test_generator
    """
    print("Loading and preprocessing data...")
    
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} does not exist")
        return None, None, None
    
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')
    
    if not os.path.exists(train_dir):
        print(f"Error: Train directory {train_dir} does not exist")
        return None, None, None
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} does not exist")
        return None, None, None
    
    try:
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split,  # Split training data for validation
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for test data
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Flow from directory with validation split
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48, 48),
            color_mode='grayscale',
            batch_size=32,
            class_mode='categorical',
            subset='training',  # Specify training subset
            shuffle=True
        )
        
        validation_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48, 48),
            color_mode='grayscale',
            batch_size=32,
            class_mode='categorical',
            subset='validation',  # Specify validation subset
            shuffle=False
        )
        
        # Test data
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(48, 48),
            color_mode='grayscale',
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"Found {train_generator.samples} training images")
        print(f"Found {validation_generator.samples} validation images")
        print(f"Found {test_generator.samples} test images")
        
        return train_generator, validation_generator, test_generator
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def train_model(train_generator, validation_generator, test_generator, 
                epochs=50, batch_size=64, output_path='emotion_model.h5'):
    """
    Huấn luyện mô hình nhận diện cảm xúc.
    Trains the emotion recognition model.
    
    Args:
        train_generator: Bộ sinh dữ liệu huấn luyện. Training data generator.
        validation_generator: Bộ sinh dữ liệu kiểm định. Validation data generator.
        test_generator: Bộ sinh dữ liệu kiểm tra. Test data generator.
        epochs (int): Số epochs. Number of epochs.
        batch_size (int): Kích thước batch. Batch size.
        output_path (str): Đường dẫn lưu mô hình. Model save path.
    
    Returns:
        model: Mô hình đã huấn luyện. Trained model.
        history: Lịch sử huấn luyện. Training history.
    """
    print(f"Training model with {epochs} epochs and batch size {batch_size}...")
    print("Using depthwise separable convolutions for low latency")
    
    # Create model with appropriate input shape
    input_shape = (48, 48, 1)  # FER2013 standard
    model = create_model(input_shape)
    
    print(model.summary())
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        'emotion_model_checkpoint.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=callbacks,
        batch_size=batch_size
    )
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save model
    try:
        # Lưu mô hình
        model.save(output_path)
        print(f"Model saved to {output_path}")
    except PermissionError:
        # Nếu không thể lưu vào đường dẫn đã chỉ định, thử lưu vào thư mục hiện tại
        alternate_path = os.path.join('.', os.path.basename(output_path))
        print(f"Permission denied. Saving model to {alternate_path} instead")
        model.save(alternate_path)
    
    return model, history

def main():
    """Hàm chính / Main function"""
    parser = argparse.ArgumentParser(description='Train facial emotion recognition model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to FER2013 dataset with train and test folders')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--output', type=str, default='emotion_model.h5', help='Output model path')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Validation split ratio from training data')
    
    args = parser.parse_args()
    
    # Enable mixed precision for faster training on compatible GPUs
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled")
    except:
        print("Mixed precision not available, using default precision")
    
    # Load data with validation split from training data
    train_generator, validation_generator, test_generator = load_data(
        args.data_path, validation_split=args.validation_split
    )
    
    if train_generator is None or validation_generator is None:
        print("Failed to load data. Exiting.")
        return
    
    # Train model
    model, history = train_model(
        train_generator,
        validation_generator,
        test_generator,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_path=args.output
    )

if __name__ == "__main__":
    main()