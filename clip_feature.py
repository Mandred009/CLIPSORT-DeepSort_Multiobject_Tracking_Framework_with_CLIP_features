""" Script for extracting CLIP features from image detections. """

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import clip

# Class to extract features using CLIP model
class CLIPFeatureExtractor(nn.Module):
    def __init__(self, model_name='ViT-L/14'):
        super(CLIPFeatureExtractor, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.feature_dim = 512
    
    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x)
            features = nn.functional.normalize(features.float(), p=2, dim=1)
        return features

    # Input is a numpy array representing the detection
    def extract_features_from_image(self, detection):
        if detection is None or detection.size == 0:
            return np.zeros(self.feature_dim)
        
        try:
            detection = Image.fromarray(detection)
            image = self.preprocess(detection).unsqueeze(0).to(self.device)
            features = self.forward(image)
            return features.cpu().numpy().flatten()
        except Exception as e:
            return np.zeros(self.feature_dim)