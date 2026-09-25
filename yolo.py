"""Object detection backends: Ultralytics YOLO / RT-DETR, or ByteTrack YOLOX."""

from ultralytics import YOLO
import torch

from yolox_mot import YOLOXDetector, is_yolox_backbone


def build_detector(model_name, object_name="person", confidence_threshold=0.5,
                   image_size=640, test_size=None):
    """Return a detector with detect(frame) -> [[x1,y1,x2,y2], ...]."""
    if is_yolox_backbone(model_name):
        return YOLOXDetector(
            model_name,
            object_name,
            confidence_threshold,
            image_size=image_size,
            test_size=test_size,
        )
    return YOLODetector(model_name, object_name, confidence_threshold, image_size)


class YOLODetector:
    def __init__(self, model_name="yolo11l.pt", object_name="person",
                 confidence_threshold=0.5, image_size=640):
        self.model = YOLO(model_name)
        self.object_name = object_name
        self.confidence_threshold = confidence_threshold
        self.image_size = int(image_size)

    def detect(self, frame):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = self.model(
            frame,
            verbose=False,
            device=device,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
        )
        bboxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = box.cls[0]
                if self.model.names[int(cls)] == self.object_name and conf >= self.confidence_threshold:
                    bboxes.append([int(x1), int(y1), int(x2), int(y2)])
        return bboxes
    

