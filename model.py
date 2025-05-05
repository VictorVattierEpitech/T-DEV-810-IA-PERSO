# model.py

import torch
import torch.nn as nn
import torchvision.models as models
from config import pathology_labels

class CheXpertModel(nn.Module):
    """
    DenseNet-121 backbone + Global Pooling
    + Dropout → Linear Head → Sigmoid
    pour classification multi-label sur CheXpert.
    """
    def __init__(self,
                 dropout_prob: float = 0.5):
        super(CheXpertModel, self).__init__()
        backbone = models.densenet121(pretrained=True)
        self.features = backbone.features
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        num_classes = len(pathology_labels)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(backbone.classifier.in_features, num_classes)
        
    def forward(self, x: torch.Tensor):
        """
        Input:
            x (B, 3, H, W)  — images CheXpert
        Returns:
            dict label → (B, 1) probability
        """
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logits = self.classifier(x)
        probs  = torch.sigmoid(logits)
        
        return {
            label: probs[:, idx].unsqueeze(1)
            for idx, label in enumerate(pathology_labels)
        }
