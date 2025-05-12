import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights
import config

class DenseNet121Classifier(nn.Module):
    def __init__(self, num_classes=len(config.CLASSES)):
        super().__init__()
        backbone = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        num_ftrs = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_ftrs // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(num_ftrs // 2, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)


def get_model():
    return DenseNet121Classifier()