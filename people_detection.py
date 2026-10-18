import sys
import cv2
from ultralytics import YOLO
INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output.mp4"
CONF_THRESHOLD = 0.5 
PERSON_CLASS = 0       
model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    sys.exit(f"couldnt find video: {INPUT_VIDEO}")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
writer = cv2.VideoWriter(
    OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

frame_count = 0
while True:
    ok, frame = cap.read()
    if not ok:         
        break
    results = model(frame, classes=[PERSON_CLASS], conf=CONF_THRESHOLD, verbose=False)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])   
        conf = float(box.conf[0])               
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, f"person {conf:.2f}", (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    writer.write(frame)
    frame_count += 1
cap.release()
writer.release()
print(f"Done. Processed {frame_count} frames. Saved to {OUTPUT_VIDEO}")