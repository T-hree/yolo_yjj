from ultralytics import YOLO

# Load a model
model = YOLO("ultralytics/cfg/models/11/yolo11-spd.yaml")  # build a new model from YAML

# Train the model
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
