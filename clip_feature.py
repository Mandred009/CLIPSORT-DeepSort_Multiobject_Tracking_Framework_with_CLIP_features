"""CLIP appearance features for DeepSORT."""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import clip


def _as_rgb_pil(detection):
    if detection is None or getattr(detection, "size", 0) == 0:
        return None
    if detection.ndim == 3 and detection.shape[2] == 3:
        detection = detection[:, :, ::-1]
    return Image.fromarray(detection)


class CLIPFeatureExtractor(nn.Module):
    def __init__(self, model_name="ViT-L/14"):
        super(CLIPFeatureExtractor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.feature_dim = int(self.model.visual.output_dim)

    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x)
            features = nn.functional.normalize(features.float(), p=2, dim=1)
        return features

    def extract_features_from_images(self, detections):
        results = [np.zeros(self.feature_dim, dtype=np.float32) for _ in detections]
        tensors, idxs = [], []
        for i, detection in enumerate(detections):
            image = _as_rgb_pil(detection)
            if image is None:
                continue
            try:
                tensors.append(self.preprocess(image))
                idxs.append(i)
            except Exception:
                continue
        if not tensors:
            return results
        batch = torch.stack(tensors).to(self.device)
        features = self.forward(batch).cpu().numpy()
        for i, feat in zip(idxs, features):
            results[i] = feat.astype(np.float32).ravel()
        return results

    def extract_features_from_image(self, detection):
        return self.extract_features_from_images([detection])[0]
