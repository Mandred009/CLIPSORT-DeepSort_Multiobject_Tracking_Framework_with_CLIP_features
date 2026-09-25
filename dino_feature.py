"""DINOv2 appearance features for DeepSORT."""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np


def _as_rgb_pil(detection):
    if detection is None or getattr(detection, "size", 0) == 0:
        return None
    if detection.ndim == 3 and detection.shape[2] == 3:
        detection = detection[:, :, ::-1]
    return Image.fromarray(detection)


class DINOv2FeatureExtractor(nn.Module):
    def __init__(self, model_name="dinov2_vitb14"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.to(self.device)
        self.model.eval()
        self.feature_dim = 768

        self.transform = transforms.Compose([
            transforms.Resize((252, 126)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features_from_images(self, detections):
        results = [np.zeros(self.feature_dim, dtype=np.float32) for _ in detections]
        tensors, idxs = [], []
        for i, detection in enumerate(detections):
            image = _as_rgb_pil(detection)
            if image is None:
                continue
            try:
                tensors.append(self.transform(image))
                idxs.append(i)
            except Exception:
                continue
        if not tensors:
            return results
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            features = nn.functional.normalize(self.model(batch), p=2, dim=1)
        features = features.cpu().numpy()
        for i, feat in zip(idxs, features):
            results[i] = feat.astype(np.float32).ravel()
        return results

    def extract_features_from_image(self, detection):
        return self.extract_features_from_images([detection])[0]
