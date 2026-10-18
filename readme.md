# Week 2: Person Detection with YOLO

## Overview
This project runs a pretrained YOLO11 model (`yolo11n.pt`) on a video to
detect people. Only the "person" class (class 0 in the COCO dataset) is kept.
Each person is drawn with a bounding box and a confidence score, and the
annotated video is saved as `output.mp4`.

## Model and Dataset
- **Model:** YOLO11 nano (Ultralytics), pretrained
- **Pretrained on:** COCO dataset (80 classes, person = class 0)
- **Confidence threshold:** 0.5

## How It Works
1. Load the pretrained YOLO model.
2. Read the video frame by frame with OpenCV.
3. Run detection on each frame, keeping only people above the threshold.
4. Draw boxes and confidence scores on the frame.
5. Write every frame to `output.mp4`.

## How to Run
1. Install the libraries:

       pip install ultralytics opencv-python

2. Put your video in this folder and name it `input.mp4`
   (or pass a different path as an argument).
3. Run:

       python detect_people.py

   or with a custom video:

       python detect_people.py my_video.mp4

4. The result is saved as `output.mp4`.
