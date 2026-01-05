""" Script to get Bounding Boxes using YOLO model. """

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Class to perform object detection using YOLO
class YOLODetector:
    def __init__(self, model_name="yolo11l.pt", object_name="person", confidence_threshold=0.5):
        self.model = YOLO(model_name)
        self.object_name = object_name
        self.confidence_threshold = confidence_threshold

    def detect(self, frame):
        device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
        results = self.model(frame, verbose=False, device=device)
        bboxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = box.cls[0]
                if self.model.names[int(cls)] == self.object_name and conf >= self.confidence_threshold:
                    bboxes.append([int(x1), int(y1), int(x2), int(y2)])
        return bboxes
    

