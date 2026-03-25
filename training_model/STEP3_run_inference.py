"""Test the model on new images and obtain the predicted labels and centroids of detected objects
"""
from ultralytics import YOLO

# Confg
MODEL_PATH = "runs/detect/train/weights/best.pt"
CONF_THRESHOLD = 0.7
IMAGE_PATH = "data/sample_images/test.jpg"
OUTPUT_PATH = "output.jpg"

def get_centroid(xyxy):
    x1, y1, x2, y2 = map(int, xyxy[0])
    return ( (x1 + x2) // 2, (y1 + y2) // 2 )


def run_inference():
    model = YOLO(MODEL_PATH)
    results = model(IMAGE_PATH)

    results[0].save(OUTPUT_PATH)

    label_info = {}

    for result in results:
        for nr in range(len(result.boxes)):
            conf = float(result.boxes[nr].conf)

            if conf > CONF_THRESHOLD:
                cls = int(result.boxes[nr].cls)
                name = model.names[cls]
                xyxy = result.boxes[nr].xyxy

                x, y = get_centroid(xyxy)
                label_info[nr] = {"label": name, "centroid": (x, y)}

    print(label_info)


if __name__ == "__main__":
    run_inference()