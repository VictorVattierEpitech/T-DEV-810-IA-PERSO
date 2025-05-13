import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import config
from data import get_dataloaders
from model import get_model

# ici on crée une fonction de perte spéciale qui ignore les labels manquants
# on utilise une focal loss qui aide à mieux apprendre sur les classes rares
class MaskedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=config.FOCAL_GAMMA, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, mask):
        # on calcule une perte binaire sans réduction
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, weight=self.alpha, reduction='none'
        )
        # focal loss applique un poids plus fort sur les erreurs
        pt = torch.exp(-bce)
        loss = (1 - pt) ** self.gamma * bce
        # on ignore les valeurs où le masque vaut zéro
        loss = loss * mask
        # si on veut la moyenne on normalise par le nombre de labels valides
        if self.reduction == 'mean':
            return loss.sum() / (mask.sum() + 1e-6)
        else:
            return loss.sum()

# ici on prépare les dossiers de sortie pour stocker les résultats et les logs
OUTPUT_DIR = 'outputs'
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'train.log')),
        logging.StreamHandler()
    ]
)

# on choisit la carte graphique si elle est dispo sinon le processeur
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# on récupère le modèle et on le met sur le bon device
model = get_model().to(device)

# on charge les données d'entraînement et de validation
_, val_loader, _, train_ds = get_dataloaders()

# === ici on met en place un suréchantillonnage ciblé pour aider le modèle à voir plus souvent les classes rares ===

# on récupère les labels et le masque du dataset en numpy
labels_np = train_ds.labels.numpy()   # chaque ligne est une image chaque colonne une classe
mask_np   = train_ds.mask.numpy()     # 1 si le label est présent 0 sinon

# on définit les classes rares à suréchantillonner avec un facteur multiplicatif
oversample_factors = {
    "Atelectasis":               5,
    "Pneumothorax":               5.0,
    "Enlarged Cardiomediastinum": 5.0,
}

# on commence avec des poids égaux pour tous les exemples
N = len(train_ds)
sample_weights = np.ones(N, dtype=np.float32)

# on augmente les poids des exemples positifs valides pour les classes rares
for cls, factor in oversample_factors.items():
    j = config.CLASSES.index(cls)
    positives = (labels_np[:, j] == 1) & (mask_np[:, j] == 1)
    sample_weights[positives] *= factor

# on crée un sampler qui va tirer les exemples en fonction de leur poids
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=N,
    replacement=True
)

# on crée le data loader avec le sampler au lieu du shuffle classique
train_loader = DataLoader(
    train_ds,
    batch_size=config.BATCH_SIZE,
    sampler=sampler,
    num_workers=config.NUM_WORKERS,
    pin_memory=True
)

# on calcule le nombre de positifs et négatifs valides pour chaque classe
pos = ((labels_np == 1) & (mask_np == 1)).sum(axis=0)
neg = ((labels_np == 0) & (mask_np == 1)).sum(axis=0)
# on crée un poids par classe basé sur le ratio négatif sur positif
pos_weight = torch.tensor(neg / (pos + 1e-6), dtype=torch.float32, device=device)

# on définit la fonction de perte avec les poids calculés
criterion = MaskedFocalLoss(alpha=pos_weight)
# on choisit l’optimiseur adamw avec le taux d’apprentissage et le weight decay
optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

# on prépare le scheduler pour gérer automatiquement le taux d’apprentissage
steps = len(train_loader)
scheduler = OneCycleLR(
    optimizer, max_lr=config.MAX_LR, steps_per_epoch=steps,
    epochs=config.EPOCHS, pct_start=0.3, anneal_strategy='cos'
)

# on prépare un dictionnaire pour stocker les métriques au fil des époques
history = {'train_loss': [], 'val_loss': [], 'acc': [], 'prec': [], 'rec': [], 'f1': []}
best_val_loss = float('inf')

# on lance l'entraînement
logging.info('🚀 Début de l\'entraînement!')
for epoch in range(1, config.EPOCHS + 1):
    t0 = time.time()
    model.train()
    train_loss = 0
    # on fait une boucle sur les batches du train loader
    for imgs, labels, mask in tqdm(train_loader, desc=f'Train Ép{epoch}/{config.EPOCHS}'):
        imgs, labels, mask = imgs.to(device), labels.to(device), mask.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels, mask)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_ds)

    # maintenant on passe en mode évaluation sans gradients
    model.eval()
    val_loss = 0; preds, trues, masks = [], [], []
    with torch.no_grad():
        for imgs, labels, mask in tqdm(val_loader, desc=f'Val   Ép{epoch}/{config.EPOCHS}'):
            imgs = imgs.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels.to(device), mask.to(device))
            val_loss += loss.item() * imgs.size(0)
            preds.append(torch.sigmoid(outputs).cpu().numpy())
            trues.append(labels.numpy())
            masks.append(mask.numpy())
    val_loss /= len(val_loader.dataset)

    # on aplatit les prédictions et les vrais labels et on applique le masque
    preds_np = np.vstack(preds) >= 0.5
    trues_np = np.vstack(trues) >= 0.5
    mask_flat = np.vstack(masks).flatten().astype(bool)
    p_flat = preds_np.flatten()[mask_flat]
    t_flat = trues_np.flatten()[mask_flat]

    # on calcule les métriques classiques
    acc = (p_flat == t_flat).mean()
    from sklearn.metrics import precision_score, recall_score, f1_score
    prec = precision_score(t_flat, p_flat, average='macro', zero_division=0)
    rec  = recall_score   (t_flat, p_flat, average='macro', zero_division=0)
    f1   = f1_score       (t_flat, p_flat, average='macro', zero_division=0)

    # on ajoute les résultats dans l’historique
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['acc'].append(acc)
    history['prec'].append(prec)
    history['rec'].append(rec)
    history['f1'].append(f1)

    # on affiche les résultats dans les logs
    logging.info(
        f"Ép{epoch} | train {train_loss:.4f} | val {val_loss:.4f} | "
        f"Acc {acc:.4f} | Prec {prec:.4f} | Rec {rec:.4f} | F1 {f1:.4f} | Durée {(time.time()-t0):.1f}s"
    )
    # si le modèle est meilleur que le précédent on le sauvegarde
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
        logging.info('🏆 Meilleur modèle sauvegardé')

# à la fin on sauvegarde aussi la dernière version du modèle
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'last_model.pth'))
logging.info('✅ Entraînement terminé!')

# on trace les courbes d'évolution des métriques et on les sauvegarde
import matplotlib.pyplot as plt
for k, v in history.items():
    plt.figure()
    plt.plot(range(1, config.EPOCHS + 1), v)
    plt.title(k)
    plt.xlabel('Ép.')
    plt.ylabel(k)
    plt.savefig(os.path.join(FIGURES_DIR, f"{k}.png"))
    plt.close()
