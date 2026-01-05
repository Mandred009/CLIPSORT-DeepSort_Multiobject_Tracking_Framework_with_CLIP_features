""" Script for extracting Dino features from image detections. """

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class DINOv2FeatureExtractor(nn.Module):
    def __init__(self, model_name='dinov2_vitb14'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.to(self.device)
        self.model.eval()
        self.feature_dim = 768  # Feature dimension for DINOv2 ViT-B/14
        
        self.transform = transforms.Compose([
            transforms.Resize((252, 126)),  # Multiple of 14
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def extract_features_from_image(self, detection):
        if detection is None or detection.size == 0:
            return np.zeros(self.feature_dim)
        try:
            image = Image.fromarray(detection)
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(image)
                features = nn.functional.normalize(features, p=2, dim=1)
            return features.cpu().numpy().flatten()
        except:
            return np.zeros(self.feature_dim)