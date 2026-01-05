""" Deep SORT implementation for multi-object tracking. """

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from resnet_feature import ResNetFeatureExtractor
from clip_feature import CLIPFeatureExtractor
from dino_feature import DINOv2FeatureExtractor
from kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# Dataclass to hold track information
@dataclass
class Track:
    track_id: int
    bbox: list
    feature: deque = field(default_factory=lambda: deque(maxlen=500)) # this is because if we use only deque it will be shared among all instances
    age: int = 1
    time_since_last_update: int = 0
    hits: int = 1
    kalman_filter: KalmanFilter=None
    is_confirmed: bool=False

# Class to perform Deep SORT tracking
class DeepSort:
    def __init__(self, max_age=30, min_hits=3, feature_extractor="resnet", mahalanobis_threshold=9.4877,
                 cosine_threshold=0.3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 1
        self.mahalanobis_threshold = mahalanobis_threshold
        self.cosine_threshold = cosine_threshold
        self.iou_threshold = iou_threshold

        # Select the feature extractor
        if feature_extractor == "resnet":
            self.feature_extractor = ResNetFeatureExtractor()
        elif feature_extractor == "clip":
            self.feature_extractor = CLIPFeatureExtractor()
        elif feature_extractor == "dino":
            self.feature_extractor = DINOv2FeatureExtractor()
        else:
            self.feature_extractor = None
    
    # Add a new track
    def add_track(self, bbox, frame):
        feature = self.feature_extractor.extract_features_from_image(frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])
        init_state=self.bbox_to_state(bbox)
        kf=KalmanFilter(initial_state=init_state)
        track = Track(track_id=self.next_id, bbox=bbox, kalman_filter=kf)
        track.feature.append(feature)
        self.tracks.append(track)
        self.next_id += 1
    
    # Convert bounding box to Kalman filter state
    def bbox_to_state(self, bbox):
        u = (bbox[0] + bbox[2]) / 2.0
        v = (bbox[1] + bbox[3]) / 2.0
        h = bbox[3] - bbox[1]
        w = bbox[2] - bbox[0]
        y = w / h
        return np.array([u, v, y, h, 0, 0, 0, 0]).reshape(-1, 1)
    
    # Convert bounding box to measurement for Kalman filter
    def bbox_to_measurement(self, bbox):
        u = (bbox[0] + bbox[2]) / 2.0
        v = (bbox[1] + bbox[3]) / 2.0
        h = bbox[3] - bbox[1]
        w = bbox[2] - bbox[0]
        y = w / h
        return np.array([u, v, y, h]).reshape(-1, 1)

    # Convert Kalman filter state to bounding box
    def state_to_bbox(self, state):
        u, v, y, h = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
        w = y * h
        x1 = int(u - w / 2)
        y1 = int(v - h / 2)
        x2 = int(u + w / 2)
        y2 = int(v + h / 2)
        return [x1, y1, x2, y2]

    # Update tracks with new detections
    def update(self, detections, frame):

        # Predict new locations of existing tracks
        for track in self.tracks:
            track.kalman_filter.predict()
            track.bbox = self.state_to_bbox(track.kalman_filter.state)  # Update predicted bbox
            track.time_since_last_update += 1
            track.age += 1

        # Perform cascade matching
        matches, unmatched_detections, unmatched_tracks = self.cascade_matching(detections, frame)

        # Update matched tracks
        for track, det in matches:
            measurement = self.bbox_to_measurement(det)
            track.kalman_filter.update(measurement)
            track.bbox = det
            feature_det = self.feature_extractor.extract_features_from_image(frame[det[1]:det[3], det[0]:det[2]])
            track.feature.append(feature_det)
            track.time_since_last_update = 0
            track.hits += 1
            if not track.is_confirmed and track.hits >= self.min_hits:
                track.is_confirmed = True
        
        # Create new tracks for unmatched detections
        for det in unmatched_detections:
            self.add_track(det, frame)

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if not (t.time_since_last_update > self.max_age)]

        # Return confirmed tracks
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed and t.time_since_last_update == 0]

        return confirmed_tracks

    # Perform cascade matching
    def cascade_matching(self, detections, frame):
        matches=[]
        unmatched_detections=detections.copy()
        unmatched_tracks=self.tracks.copy()

        for a in range(1,self.max_age+1):
            if len(unmatched_detections)==0 or len(unmatched_tracks)==0:
                break
            
            # Select tracks according to increasing order of age
            tracks_at_age=[t for t in unmatched_tracks if t.time_since_last_update==a]
            if len(tracks_at_age)==0:
                continue

            cost_matrix=np.full((len(tracks_at_age), len(unmatched_detections)), 1e9)

            for t_idx,t in enumerate(tracks_at_age):
                for det_idx,det in enumerate(unmatched_detections):
                    measurement=self.bbox_to_measurement(det)
                    dist=t.kalman_filter.mahalanobis_distance(measurement)

                    cos_dist=1e9
                    if dist<self.mahalanobis_threshold:
                        feature_det=self.feature_extractor.extract_features_from_image(frame[det[1]:det[3], det[0]:det[2]])
        
                        for feat in t.feature:
                            cos_dist=min(cos_dist, self.get_cosine_distance(feat, feature_det))

                    cost_matrix[t_idx, det_idx]=cos_dist

            # Hungarian assignment for cosine cost matrix        
            row_ind, col_ind=self.hungarian_assignment(cost_matrix)
            matched_det_indices = set()
            matched_track_indices = set()
            
            # Record matches below the cosine threshold
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < self.cosine_threshold:
                    matches.append((tracks_at_age[r], unmatched_detections[c]))
                    matched_det_indices.add(c)
                    matched_track_indices.add(r)

            matched_tracks = [tracks_at_age[i] for i in matched_track_indices]
            unmatched_tracks = [t for t in unmatched_tracks if t not in matched_tracks]
            
            unmatched_detections = [d for i, d in enumerate(unmatched_detections) 
                                    if i not in matched_det_indices]
            
            if a==1 and unmatched_detections and unmatched_tracks:
                # Get only age-1 tracks that failed appearance matching
                unmatched_age1_tracks = [t for t in tracks_at_age if t not in matched_tracks]
                
                if unmatched_age1_tracks:
                    iou_cost_matrix=np.full((len(unmatched_age1_tracks), len(unmatched_detections)), 1e9)
                    for t_idx, t in enumerate(unmatched_age1_tracks):
                        for det_idx, det in enumerate(unmatched_detections):
                            iou=self.iou_score(t.bbox, det)
                            iou_cost_matrix[t_idx, det_idx]=1-iou
                    
                    # Hungarian assignment for IoU cost matrix
                    row_ind, col_ind=self.hungarian_assignment(iou_cost_matrix)
                    matched_det_indices = set()
                    matched_track_indices = set()

                    for r, c in zip(row_ind, col_ind):
                        if iou_cost_matrix[r, c] < (1 - self.iou_threshold):
                            matches.append((unmatched_age1_tracks[r], unmatched_detections[c]))
                            matched_det_indices.add(c)
                            matched_track_indices.add(r)

                    matched_tracks = [unmatched_age1_tracks[i] for i in matched_track_indices]
                    unmatched_tracks = [t for t in unmatched_tracks if t not in matched_tracks]
                    unmatched_detections = [d for i, d in enumerate(unmatched_detections)
                                            if i not in matched_det_indices]

                            
        return matches, unmatched_detections, unmatched_tracks

    # Compute cosine distance between two feature vectors
    def get_cosine_distance(self, feature1, feature2)->float:
        feature1=feature1.flatten()
        feature2=feature2.flatten()

        dot_product = np.dot(feature1, feature2)
        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)
        if( norm1==0 or norm2==0):
            return 1.0
 
        return 1 - (dot_product / (norm1 * norm2))
    
    # Function to perform Hungarian assignment
    def hungarian_assignment(self, cost_matrix):
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind, col_ind
    
    # Compute IoU score between two bounding boxes
    def iou_score(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0
    