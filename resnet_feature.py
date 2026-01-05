""" Resnet feature extractor for DeepSORT. """
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Resnet class for feature extraction as a baseline
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super(ResNetFeatureExtractor, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT if model_name == 'resnet50' else 'DEFAULT'
            self.model = getattr(models, model_name)(weights=weights)
        else:
            self.model = getattr(models, model_name)(weights=None)
        
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove classification layer
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
            features = nn.functional.normalize(features, p=2, dim=1)  # L2 normalize
        return features

    # Input is a numpy array representing the detection
    def extract_features_from_image(self, detection):
        if detection is None or detection.size == 0:
            return np.zeros(self.feature_dim)
        
        try:
            detection = Image.fromarray(detection)
            image = self.transform(detection).unsqueeze(0).to(self.device)
            features = self.forward(image)
            return features.cpu().numpy().flatten()
        except Exception as e:
            return np.zeros(self.feature_dim)