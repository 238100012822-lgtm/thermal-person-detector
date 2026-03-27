Thermal Image Person Detection

This project implements a thermal image person detection system using pre-trained YOLOv8 models.
The system detects human subjects in thermal images and evaluates performance under different confidence thresholds.

Project Description

Thermal imaging enables human detection in low-light and night-time conditions by capturing infrared radiation instead of visible light.
This project applies deep learning-based object detection using YOLOv8 to identify persons in thermal images.

Models Compared

Two YOLOv8 variants were evaluated:

YOLOv8n (Nano version – lightweight and fast)
YOLOv8s (Small version – larger and more accurate)
Confidence Thresholds Tested

The following confidence values were evaluated:

0.15
0.25
0.50

Performance was analyzed by comparing total detections across thermal test images.

Final Model Selection

YOLOv8n was selected as the final model due to its balanced performance, speed, and lower computational requirements.

Requirements

Install required libraries:

pip install ultralytics opencv-python

How to Run
Place your thermal images inside the "my_image" folder.
Run the script:

python Thermal Image Person Detector.py

Detection results will be saved in the "results" folder.
Project Structure

thermal-person-detector/
│
├── Thermal Image Person Detector.py
├── results/
└── README.md

Technologies Used
Python
YOLOv8 (Ultralytics)
OpenCV
Deep Learning
