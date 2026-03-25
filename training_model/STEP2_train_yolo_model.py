"""Train the model using dataset (yaml) and the YOLOv8 architecture"""

from ultralytics import YOLO

# Config
YAML_PATH = "data.yaml"
BASE_MODEL = "yolov8n.pt"

def train():
    model = YOLO(BASE_MODEL)
    model.train(
        data=YAML_PATH,
        epochs=40,
        imgsz=640,
        batch=8,
        patience=15
    )

if __name__ == "__main__":
    train()