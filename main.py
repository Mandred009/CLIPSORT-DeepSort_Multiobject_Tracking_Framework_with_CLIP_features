""" Main script to run object detection and tracking on a video using YOLO and DeepSORT."""

import os
import random
import time
import yaml
import cv2
from yolo import YOLODetector
from deepsort import DeepSort
from collections import deque
from dataclasses import dataclass, field

# Dataclass to hold trajectory information for visualization
@dataclass
class TrackTrajectory:
    track_id: int
    trajectory: deque = field(default_factory=lambda: deque(maxlen=10))
    color: tuple = field(default_factory=lambda: (
        random.randint(100, 200), 
        random.randint(100, 200), 
        random.randint(100, 200)
    ))

# Load configuration from YAML file
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Function to visualize trajectories
def visualize_trajectories(frame, trajectories, active_track_ids=None):

    for track_id, traj in trajectories.items():
        points = list(traj.trajectory)
        if(active_track_ids is not None and track_id not in active_track_ids):
            continue  # Skip inactive tracks

        # Draw trajectory as connected lines
        for i in range(1, len(points)):
            alpha = i / len(points)  # 0 to 1
            thickness = max(1, int(2 * alpha))
            cv2.line(frame, points[i-1], points[i], traj.color, thickness)
        
        # Draw current position as a larger dot
        if points:
            cv2.circle(frame, points[-1], 4, traj.color, -1)


if __name__ == "__main__":
    config = load_config('config.yaml')
    print(config)

    # Config parameters
    detection_model_name = config['params']['detection_backbone']
    feature_extractor_model = config['params']['feature_extractor']
    tracked_entity = config['params']['tracked_entity']
    max_age = config['params']['max_age']
    min_hits = config['params']['min_hits']
    detection_confidence_threshold = config['params']['detection_confidence_threshold']
    mahalanobis_threshold = config['params']['mahalanobis_threshold']
    cosine_threshold = config['params']['cosine_threshold']
    iou_threshold = config['params']['iou_threshold']
    save_results_loc = config['params']['save_results_loc']
    visualize_traj = config['params'].get('visualize_traj', "false").lower() == "true"

    os.makedirs(save_results_loc, exist_ok=True)

    detection_model = YOLODetector(detection_model_name, tracked_entity, detection_confidence_threshold)
    deepsort_tracker = DeepSort(max_age, min_hits, feature_extractor_model, mahalanobis_threshold, cosine_threshold, iou_threshold)

    # Video path for testing
    video_path = "Test Videos\\People1.webm"

    cap = cv2.VideoCapture(video_path)
    
    # Get original video properties
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    output_size = (1800, 900)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f'{tracked_entity}_{timestamp}_{feature_extractor_model}.avi'
    output_path = os.path.join(save_results_loc, output_filename)
    
    cap2 = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'XVID'),
        orig_fps,
        output_size
    )
    
    # Dictionary to hold trajectory history
    trajectory_history = {}  # track_id -> TrackTrajectory

    start_time=time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        bboxes = detection_model.detect(frame)
        tracks = deepsort_tracker.update(bboxes, frame)

        active_track_ids = set()
        
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            track_id = track.track_id
            active_track_ids.add(track_id)
            
            # Calculate center point
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Add to trajectory history (create if doesn't exist)
            if track_id not in trajectory_history:
                trajectory_history[track_id] = TrackTrajectory(track_id=track_id)
            
            trajectory_history[track_id].trajectory.append(center)

            # Draw bounding box and ID
            if visualize_traj:
                color = trajectory_history[track_id].color
            else:
                color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Visualize trajectories
        if visualize_traj:
            visualize_trajectories(frame, trajectory_history, active_track_ids)

        frame = cv2.resize(frame, output_size)
        cap2.write(frame)
        cv2.imshow("Tracking", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    end_time=time.time()

    print(f"Total Time taken: {(end_time-start_time)/60} min")

    cap.release()
    cap2.release()
    cv2.destroyAllWindows()