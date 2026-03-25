import object_recognition

YOLO_MODEL_PATH = "models\\yolo8n_version3.pt"  # Path to your trained YOLO model weights
IMAGE_PATH = ""  # Path to your input image
OUTPUT_IMAGE_PATH = "data\\annotations\\blurred_output.jpg"  # Path to save the output image with annotations

def main():
    yolo = object_recognition.Yolo_model(conf_score=0.6)
    yolo.load_model(model=YOLO_MODEL_PATH)
    yolo.load_image(image=IMAGE_PATH, save_as=OUTPUT_IMAGE_PATH)
    labels = yolo.request_labels()
    print(labels)

if __name__ == "__main__":
    main()
