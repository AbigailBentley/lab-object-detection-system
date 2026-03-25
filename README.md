# 🧪 Lab Object Detection System (RGB-D + YOLOv8)

Computer vision system for detecting chemists and laboratory equipment, with depth-based distance estimation using an Intel RealSense camera.

---

## 📌 Overview

This project is an end-to-end computer vision pipeline designed to detect objects in laboratory environments and analyse their spatial relationships.

The system combines RGB images and depth data to enable real-world distance measurements between detected objects.

It was developed as part of a research project exploring how computer vision can support laboratory safety and automation.

---

## 🚀 Key Features

- 📡 **Custom data collection pipeline** using Intel RealSense (RGB + depth)
- 🧠 **YOLOv8 object detection model** (fine-tuned on custom dataset)
- 📏 **Depth-based distance calculation** between detected objects
- 📊 **Model evaluation** across multiple confidence thresholds
- 🧪 **Scenario-based testing** in different lab environments
- 🔐 **Privacy-aware dataset handling** (anonymised images)

---

## 🛠️ Tech Stack

- **Python** (NumPy, OpenCV, matplotlib)
- **PyTorch / Ultralytics YOLOv8**
- **Intel RealSense SDK (pyrealsense2)**
- **Data formats:** JPG, NPY

---

## 🤝 My Contributions

This was an independent project. I was responsible for:

- Designing and implementing the **data collection pipeline**  
- Collecting and preparing a **custom RGB-D dataset**  
- Training and fine-tuning the **YOLOv8 model**  
- Developing **object detection and centroid extraction logic**  
- Implementing **depth-based distance calculations**  
- Evaluating model performance and analysing results  

---

## 📊 How It Works

1. RealSense camera captures:
   - RGB images  
   - Depth data  

2. Data collection script:
   - Saves RGB images (`.jpg`)  
   - Stores depth arrays (`.npy`)  
   - Records depth scale  

3. YOLOv8 model:
   - Detects chemists and lab equipment  
   - Outputs bounding boxes and labels  

4. Detection system:
   - Calculates object centroids  
   - Filters detections using confidence threshold  

5. Depth module:
   - Converts pixel coordinates to 3D points  
   - Computes real-world distances between objects  

---

## 📊 Model Evaluation

The model was tested across multiple confidence thresholds to optimise performance.

### Key results:
- Best performance at **confidence = 0.75 (~97.7% accuracy)**  
- Lower thresholds → more false positives  
- Higher thresholds → missed detections  

---

## 📈 Performance Visualisation



---

## 🧪 Scenario Testing

The model was evaluated in different lab setups:

- **Dense environments (near equipment):**
  - Reduced accuracy due to occlusion  

- **Spaced environments:**
  - Improved detection performance  

- **Minimal clutter:**
  - Highest accuracy (>93%)  

---

## ⚠️ Data Privacy

The original dataset contains identifiable individuals and is not included in this repository.

Example images have been **anonymised using image blurring techniques**.

The data collection and training pipeline are fully reproducible using the provided scripts.

---

## ⚙️ Requirements

**Hardware:**
- Intel RealSense RGB-D camera  

**Python packages:**
- ultralytics  
- opencv-python  
- numpy  
- matplotlib  
- pyrealsense2  

---

## 🙌 Acknowledgements

This project was completed as part of my MSc Digital Chemistry final project at the University of Liverpool.

I was a member of the Cooper Group, supervised by Dr. Satheeshkumar Veeramani and Dr. Gabriella Pizzuto. They also supported the project by participating in data collection as part of the dataset.

---

## ▶️ Usage

Run inference on a new image:

```bash
python src/run_inference.py
