import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights
import config

class DenseNet121Classifier(nn.Module):
    def __init__(self, num_classes=len(config.CLASSES)):
        super().__init__()
        # 1️⃣ Charger le backbone DenseNet-121 pré-entraîné sur ImageNet
        backbone = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        # Récupérer le nombre de features de la dernière couche du backbone
        num_ftrs = backbone.classifier.in_features

        # 2️⃣ Retirer la tête d’origine (cible ImageNet 1000 classes)
        #    pour ne garder que l’extracteur de features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # 3️⃣ Définir une nouvelle tête (head) de classification MLP adaptée à CheXpert
        #    - Dropout pour régularisation (éviter l’overfitting)
        #    - Linear pour réduire la dimensionnalité des features
        #    - ReLU pour introduire de la non-linéarité
        #    - Dropout + Linear final pour projeter vers les N classes
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),                              # 50% de neurones éteints aléatoirement
            nn.Linear(num_ftrs, num_ftrs // 2),           # réduire d’un facteur 2
            nn.ReLU(inplace=True),                        # activation simple et rapide
            nn.Dropout(0.5),                              # deuxième couche de Dropout
            nn.Linear(num_ftrs // 2, num_classes),        # sortie finale (14 pathologies)
        )

    def forward(self, x):
        # 4️⃣ Passage avant (forward pass)
        #    - x : batch d’images (B, 3, 224, 224)
        #    - feats : features extraites par DenseNet (B, num_ftrs)
        feats = self.backbone(x)
        #    - classifier(feats) : logits pour chaque pathologie (B, num_classes)
        return self.classifier(feats)

def get_model():
    """
    Factory function :
    - Permet d'instancier le modèle depuis n'importe quel script
      sans répéter le nom de la classe.
    """
    return DenseNet121Classifier()
