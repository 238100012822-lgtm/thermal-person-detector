# Thermal Image Person Detection

This project implements a thermal image person detection system using YOLOv8.

## Project Description
The system detects persons in thermal images using pre-trained YOLOv8 models.

## Models Compared
- YOLOv8n
- YOLOv8s

## Confidence Values Tested
- 0.15
- 0.25
- 0.50

## Final Model
YOLOv8n was selected due to balanced performance and speed.

## How to Run
1. Install dependencies:
   pip install ultralytics opencv-python

2. Run the script:
   python detection.py
