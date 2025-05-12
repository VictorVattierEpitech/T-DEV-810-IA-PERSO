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

# Focal loss masquée pour ignorer -1
class MaskedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=config.FOCAL_GAMMA, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, mask):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, weight=self.alpha, reduction='none'
        )
        pt = torch.exp(-bce)
        loss = (1 - pt) ** self.gamma * bce
        loss = loss * mask
        if self.reduction == 'mean':
            return loss.sum() / (mask.sum() + 1e-6)
        else:
            return loss.sum()

# Setup des dossiers et du logger
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model().to(device)

# Chargement des DataLoaders et du dataset
_, val_loader, _, train_ds = get_dataloaders()

# === OVERSAMPLING CIBLÉ ===
# 1. Récupérer labels et mask
labels_np = train_ds.labels.numpy()   # shape (N, C), valeurs {0,1}
mask_np   = train_ds.mask.numpy()     # shape (N, C), 1 si label valide

# 2. Facteurs d’oversampling pour les classes rares
oversample_factors = {
    "Atelectasis":               10.0,
    "Pneumothorax":               5.0,
    "Enlarged Cardiomediastinum": 3.0,
}

# 3. Initialiser poids uniformes
N = len(train_ds)
sample_weights = np.ones(N, dtype=np.float32)

# 4. Appliquer les facteurs aux échantillons positifs valides
for cls, factor in oversample_factors.items():
    j = config.CLASSES.index(cls)
    positives = (labels_np[:, j] == 1) & (mask_np[:, j] == 1)
    sample_weights[positives] *= factor

# 5. Construire le sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=N,
    replacement=True
)

# 6. Recréer le train_loader avec sampler
train_loader = DataLoader(
    train_ds,
    batch_size=config.BATCH_SIZE,
    sampler=sampler,
    num_workers=config.NUM_WORKERS,
    pin_memory=True
)

# Calcul des pos/neg pour la perte pondérée
pos = ((labels_np == 1) & (mask_np == 1)).sum(axis=0)
neg = ((labels_np == 0) & (mask_np == 1)).sum(axis=0)
pos_weight = torch.tensor(neg / (pos + 1e-6), dtype=torch.float32, device=device)

criterion = MaskedFocalLoss(alpha=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
steps = len(train_loader)
scheduler = OneCycleLR(
    optimizer, max_lr=config.MAX_LR, steps_per_epoch=steps,
    epochs=config.EPOCHS, pct_start=0.3, anneal_strategy='cos'
)

history = {'train_loss': [], 'val_loss': [], 'acc': [], 'prec': [], 'rec': [], 'f1': []}
best_val_loss = float('inf')

logging.info('🚀 Début de l\'entraînement!')
for epoch in range(1, config.EPOCHS + 1):
    t0 = time.time()
    model.train()
    train_loss = 0
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

    # Calcul des métriques (flatten + mask)
    preds_np = np.vstack(preds) >= 0.5
    trues_np = np.vstack(trues) >= 0.5
    mask_flat = np.vstack(masks).flatten().astype(bool)
    p_flat = preds_np.flatten()[mask_flat]
    t_flat = trues_np.flatten()[mask_flat]

    acc = (p_flat == t_flat).mean()
    from sklearn.metrics import precision_score, recall_score, f1_score
    prec = precision_score(t_flat, p_flat, average='macro', zero_division=0)
    rec  = recall_score   (t_flat, p_flat, average='macro', zero_division=0)
    f1   = f1_score       (t_flat, p_flat, average='macro', zero_division=0)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['acc'].append(acc)
    history['prec'].append(prec)
    history['rec'].append(rec)
    history['f1'].append(f1)

    logging.info(
        f"Ép{epoch} | train {train_loss:.4f} | val {val_loss:.4f} | "
        f"Acc {acc:.4f} | Prec {prec:.4f} | Rec {rec:.4f} | F1 {f1:.4f} | Durée {(time.time()-t0):.1f}s"
    )
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
        logging.info('🏆 Meilleur modèle sauvegardé')

# Save final
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'last_model.pth'))
logging.info('✅ Entraînement terminé!')

# Plot metrics
import matplotlib.pyplot as plt
for k, v in history.items():
    plt.figure()
    plt.plot(range(1, config.EPOCHS + 1), v)
    plt.title(k)
    plt.xlabel('Ép.')
    plt.ylabel(k)
    plt.savefig(os.path.join(FIGURES_DIR, f"{k}.png"))
    plt.close()
