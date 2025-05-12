# model.py

import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights
import config


class DenseNet121Classifier(nn.Module):
    def __init__(self, num_classes=len(config.CLASSES)):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT
        backbone = models.densenet121(weights=weights)
        num_ftrs = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)


def get_model():
    return DenseNet121Classifier()
