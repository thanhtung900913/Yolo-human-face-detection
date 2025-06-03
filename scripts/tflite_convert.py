from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
putenv("ROCM_PATH", "/opt/rocm-6.3.3")

import tensorflow as tf

# 1. Load và export model
model = tf.keras.models.load_model('./models/emotion_model_2.h5')
model.export('saved_model_format')

# 2. Khởi tạo converter từ SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_format')

# 3. Bật cả built‑in ops và TF Select ops
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,    # các op chuẩn TFLite
    tf.lite.OpsSet.SELECT_TF_OPS       # cho phép dùng Flex ops
]

# Bật FP16 quantization cho các weight và operation bên trong
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# 4. Convert
tflite_model = converter.convert()

# 5. Ghi ra file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Done! Model saved to model.tflite")
