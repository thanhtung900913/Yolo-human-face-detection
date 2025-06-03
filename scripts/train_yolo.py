from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
putenv("ROCM_PATH", "/opt/rocm-6.3.3")

from ultralytics import YOLO

model_yolo11n = YOLO('runs/detect/train2/weights/best.pt')

model_yolo11n_face = model_yolo11n.train(
    data='data.yml',
    epochs=100,
    batch=4,
    imgsz=640,
    seed=69,
    optimizer='NAdam',
    lr0=0.0005,
    warmup_epochs=5,
    verbose=True,
    half=True
)