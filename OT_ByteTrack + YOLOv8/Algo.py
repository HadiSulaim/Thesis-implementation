import cv2
import os
import numpy as np
from ultralytics import YOLO

class ObjectTracking:
    def __init__(self):
        # Use the current working directory
        dir = os.getcwd()
        weights_path = os.path.join(dir, 'yolov8s.pt')
        self.video_path = os.path.join(dir, 'street2.mp4')
        self.bytetrack_yaml_path = os.path.join(dir, 'bytetrack.yaml')
        self.model = YOLO(weights_path)

    def detect_object(self):
        results = self.model.predict(source=self.video_path, show=True, line_width=1)

    def track_object(self):
        frame_count = 0
        n_frames = 1
        image_scale = 1
        cap = cv2.VideoCapture(self.video_path)

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('tracked_output.mp4', fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            height, width = frame.shape[:2]
            new_width = round(width / image_scale)
            new_height = round(height / image_scale)
            frame = cv2.resize(frame, (new_width, new_height))

            frame_count += 1
            if frame_count % n_frames != 0:
                continue

            results = self.model.track(source=frame, persist=True, tracker=self.bytetrack_yaml_path)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Get class IDs
            
            for box, id, class_id in zip(boxes, ids, class_ids):
                if class_id == 0:  # Assuming class ID 0 corresponds to 'person'
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 2)
                    cv2.putText(frame, f"Id: {id}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            # Write the frame into the file 'tracked_output.mp4'
            out.write(frame)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

def run_detect_object():
    ot = ObjectTracking()
    ot.detect_object()

def run_track_object():
    ot = ObjectTracking()
    ot.track_object()

if __name__ == '__main__':
    #run_detect_object()
    run_track_object()