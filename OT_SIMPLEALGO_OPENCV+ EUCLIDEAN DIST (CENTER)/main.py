import cv2
import os
from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("street2.mp4")

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# Create output directory if it doesn't exist
output_dir = "run"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the codec and create VideoWriter object for .mp4 output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Alternatively, try 'X264' if 'mp4v' doesn't work
out = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'), fourcc, 30.0, (1280, 720))  # Adjust frame size as needed

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width, _ = frame.shape

    # Extract Region of interest
    roi = frame[0:720, 0:1280]

    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Write the frame into the file 'output.mp4'
    out.write(roi)

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()