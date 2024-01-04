# Object Detection with TensorFlow Lite

This repository contains a Python script for real-time object detection using TensorFlow Lite, a lightweight solution for deploying machine learning models on edge devices.

## Description

The Python script `main.py` leverages a pre-trained model and a label map (`coco.names`) to perform object detection on live video frames captured from the webcam. It draws bounding boxes around detected objects with labels and confidence scores.

### Features

- **Real-time Object Detection:** Utilizes the webcam for live object detection.
- **Bounding Box Visualization:** Marks detected objects with bounding boxes and labels.
- **Customizable Thresholds:** Adjustable confidence thresholds for object detection results.
- **Scalability:** Suitable for various applications needing real-time object detection on edge devices.

## Technology Stack

- **TensorFlow Lite:** Efficiently deploys machine learning models for edge devices.
- **OpenCV:** Library for image and video processing.
- **NumPy:** Scientific computing library for array manipulation and numerical operations.

## Files Included

1. **main.py:**
   - The main Python script for object detection using TensorFlow Lite.

2. **coco.names:**
   - File containing class names or labels used in object detection.

3. **requirements.txt:**
   - File listing Python dependencies required for running the object detection script.


   ```bash
   pip install -r requirements.txt

