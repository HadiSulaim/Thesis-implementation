import cv2
import os
import numpy as np
from tracker import EuclideanDistTracker  # Assuming you have a EuclideanDistTracker or similar

# Create tracker object
tracker = EuclideanDistTracker()

# Load the pre-trained MobileNet-SSD model
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# Define class labels MobileNet SSD was trained on
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Initialize video capture
cap = cv2.VideoCapture("street2.mp4")

# Create output directory if it doesn't exist
output_dir = "run"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the codec and create VideoWriter object for .mp4 output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'), fourcc, 30.0, (1920, 1080))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame to desired output size
    frame_resized = cv2.resize(frame, (1920, 1080))

    # Prepare the frame for detection
    (h, w) = frame_resized.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame_resized, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Prepare detections for tracking
    detections_list = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:  # Higher confidence threshold for better accuracy
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "person":  # Focus on tracking persons
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x_max, y_max) = box.astype("int")
                detections_list.append([x, y, x_max - x, y_max - y])

    # Update tracker with current frame detections
    boxes_ids = tracker.update(detections_list)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(frame_resized, f'ID {id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Write the frame into the file 'output.mp4'
    out.write(frame_resized)

    cv2.imshow("Frame", frame_resized)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()