"""ResNet50 appearance features for DeepSORT."""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


def _as_rgb_pil(detection):
    if detection is None or getattr(detection, "size", 0) == 0:
        return None
    if detection.ndim == 3 and detection.shape[2] == 3:
        detection = detection[:, :, ::-1]
    return Image.fromarray(detection)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, model_name="resnet50", pretrained=True):
        super(ResNetFeatureExtractor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT if model_name == "resnet50" else "DEFAULT"
            self.model = getattr(models, model_name)(weights=weights)
        else:
            self.model = getattr(models, model_name)(weights=None)

        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()
        self.feature_dim = 2048

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        with torch.no_grad():
            features = self.model(x)
            features = features.view(features.size(0), -1)
            features = nn.functional.normalize(features, p=2, dim=1)
        return features

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
        features = self.forward(batch).cpu().numpy()
        for i, feat in zip(idxs, features):
            results[i] = feat.astype(np.float32).ravel()
        return results

    def extract_features_from_image(self, detection):
        return self.extract_features_from_images([detection])[0]
