"""Deep SORT implementation for multi-object tracking."""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from resnet_feature import ResNetFeatureExtractor
from clip_feature import CLIPFeatureExtractor
from dino_feature import DINOv2FeatureExtractor
from kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    track_id: int
    bbox: list
    feature: deque = field(default_factory=lambda: deque(maxlen=100))
    age: int = 1
    time_since_last_update: int = 0
    hits: int = 1
    kalman_filter: KalmanFilter = None
    is_confirmed: bool = False


class DeepSort:
    def __init__(self, max_age=30, min_hits=3, feature_extractor="resnet",
                 mahalanobis_threshold=9.4877, cosine_threshold=0.3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 1
        self.mahalanobis_threshold = mahalanobis_threshold
        self.cosine_threshold = cosine_threshold
        self.iou_threshold = iou_threshold

        if feature_extractor == "resnet":
            self.feature_extractor = ResNetFeatureExtractor()
        elif feature_extractor == "clip":
            self.feature_extractor = CLIPFeatureExtractor()
        elif feature_extractor == "dino":
            self.feature_extractor = DINOv2FeatureExtractor()
        else:
            self.feature_extractor = None

    def reset(self):
        """Clear tracks so the same loaded models can be reused on the next sequence."""
        self.tracks = []
        self.next_id = 1

    def add_track(self, bbox, feature):
        kf = KalmanFilter(initial_state=self.bbox_to_state(bbox))
        track = Track(track_id=self.next_id, bbox=bbox, kalman_filter=kf)
        track.feature.append(feature)
        self.tracks.append(track)
        self.next_id += 1

    def bbox_to_state(self, bbox):
        u = (bbox[0] + bbox[2]) / 2.0
        v = (bbox[1] + bbox[3]) / 2.0
        h = max(float(bbox[3] - bbox[1]), 1.0)
        w = max(float(bbox[2] - bbox[0]), 1.0)
        return np.array([u, v, w / h, h, 0, 0, 0, 0], dtype=np.float64).reshape(-1, 1)

    def bbox_to_measurement(self, bbox):
        u = (bbox[0] + bbox[2]) / 2.0
        v = (bbox[1] + bbox[3]) / 2.0
        h = max(float(bbox[3] - bbox[1]), 1.0)
        w = max(float(bbox[2] - bbox[0]), 1.0)
        return np.array([u, v, w / h, h], dtype=np.float64).reshape(-1, 1)

    def state_to_bbox(self, state):
        u, v, aspect, h = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
        w = aspect * h
        return [int(u - w / 2), int(v - h / 2), int(u + w / 2), int(v + h / 2)]

    def clip_bbox(self, bbox, width, height):
        x1 = max(0, min(width - 1, int(bbox[0])))
        y1 = max(0, min(height - 1, int(bbox[1])))
        x2 = max(0, min(width, int(bbox[2])))
        y2 = max(0, min(height, int(bbox[3])))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def sanitize_detections(self, detections, frame):
        height, width = frame.shape[:2]
        cleaned = []
        for det in detections:
            clipped = self.clip_bbox(det, width, height)
            if clipped is not None:
                cleaned.append(clipped)
        return cleaned

    def crop(self, frame, bbox):
        return frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    def extract_detection_features(self, frame, detections):
        if not detections:
            return []
        crops = [self.crop(frame, det) for det in detections]
        extractor = self.feature_extractor
        if hasattr(extractor, "extract_features_from_images"):
            return extractor.extract_features_from_images(crops)
        return [extractor.extract_features_from_image(crop) for crop in crops]

    def update(self, detections, frame):
        detections = self.sanitize_detections(detections, frame)

        for track in self.tracks:
            track.kalman_filter.predict()
            track.bbox = self.state_to_bbox(track.kalman_filter.state)
            track.time_since_last_update += 1
            track.age += 1

        features = self.extract_detection_features(frame, detections)
        matches, unmatched_det_idx = self.cascade_matching(detections, features)

        for track, det_idx in matches:
            det = detections[det_idx]
            track.kalman_filter.update(self.bbox_to_measurement(det))
            track.bbox = det
            track.feature.append(features[det_idx])
            track.time_since_last_update = 0
            track.hits += 1
            if not track.is_confirmed and track.hits >= self.min_hits:
                track.is_confirmed = True

        for det_idx in unmatched_det_idx:
            self.add_track(detections[det_idx], features[det_idx])

        alive = []
        for track in self.tracks:
            if not track.is_confirmed and track.time_since_last_update > 0:
                continue
            if track.time_since_last_update > self.max_age:
                continue
            alive.append(track)
        self.tracks = alive

        # Official DeepSORT: keep a confirmed track visible for one missed frame.
        return [t for t in self.tracks if t.is_confirmed and t.time_since_last_update <= 1]

    def cascade_matching(self, detections, features):
        """Appearance cascade on confirmed tracks, then IoU for the rest."""
        matches = []
        unmatched_det_idx = list(range(len(detections)))
        unmatched_confirmed = [t for t in self.tracks if t.is_confirmed]
        unconfirmed = [t for t in self.tracks if not t.is_confirmed]

        for age in range(1, self.max_age + 1):
            if not unmatched_det_idx or not unmatched_confirmed:
                break
            tracks_at_age = [t for t in unmatched_confirmed if t.time_since_last_update == age]
            if not tracks_at_age:
                continue

            cost = np.full((len(tracks_at_age), len(unmatched_det_idx)), 1e5)
            for t_idx, track in enumerate(tracks_at_age):
                for d_idx, det_idx in enumerate(unmatched_det_idx):
                    maha = track.kalman_filter.mahalanobis_distance(
                        self.bbox_to_measurement(detections[det_idx])
                    )
                    if maha <= self.mahalanobis_threshold:
                        cost[t_idx, d_idx] = self.min_cosine_distance(track, features[det_idx])

            row_ind, col_ind = self.hungarian_assignment(cost)
            used_tracks = set()
            used_dets = set()
            for row, col in zip(row_ind, col_ind):
                if cost[row, col] < self.cosine_threshold:
                    matches.append((tracks_at_age[row], unmatched_det_idx[col]))
                    used_tracks.add(id(tracks_at_age[row]))
                    used_dets.add(col)

            unmatched_confirmed = [t for t in unmatched_confirmed if id(t) not in used_tracks]
            unmatched_det_idx = [d for i, d in enumerate(unmatched_det_idx) if i not in used_dets]

        iou_tracks = unconfirmed + [
            t for t in unmatched_confirmed if t.time_since_last_update == 1
        ]
        if iou_tracks and unmatched_det_idx:
            cost = np.full((len(iou_tracks), len(unmatched_det_idx)), 1e5)
            for t_idx, track in enumerate(iou_tracks):
                for d_idx, det_idx in enumerate(unmatched_det_idx):
                    cost[t_idx, d_idx] = 1.0 - self.iou_score(track.bbox, detections[det_idx])

            row_ind, col_ind = self.hungarian_assignment(cost)
            used_tracks = set()
            used_dets = set()
            iou_cutoff = 1.0 - self.iou_threshold
            for row, col in zip(row_ind, col_ind):
                if cost[row, col] < iou_cutoff:
                    matches.append((iou_tracks[row], unmatched_det_idx[col]))
                    used_tracks.add(id(iou_tracks[row]))
                    used_dets.add(col)
            unmatched_det_idx = [d for i, d in enumerate(unmatched_det_idx) if i not in used_dets]

        return matches, unmatched_det_idx

    def min_cosine_distance(self, track, feature):
        if not track.feature:
            return 1.0
        gallery = np.asarray(track.feature, dtype=np.float32)
        feat = np.asarray(feature, dtype=np.float32).ravel()
        feat_norm = np.linalg.norm(feat)
        if feat_norm == 0:
            return 1.0
        feat = feat / feat_norm
        norms = np.linalg.norm(gallery, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        sims = (gallery / norms) @ feat
        return float(1.0 - np.clip(sims.max(), -1.0, 1.0))

    def hungarian_assignment(self, cost_matrix):
        if cost_matrix.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        return linear_sum_assignment(cost_matrix)

    def iou_score(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
        area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
