# train.py

import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from data import get_dataloaders
from model import get_model


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=config.FOCAL_GAMMA, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none", weight=self.alpha
        )
        pt = torch.exp(-bce)
        loss = (1 - pt) ** self.gamma * bce
        return loss.mean() if self.reduction == "mean" else loss.sum()


# Setup
OUTPUT_DIR = "outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "train.log")),
        logging.StreamHandler(),
    ],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)

train_loader, val_loader, _, train_ds = get_dataloaders()

# compute pos_weight
labels = train_ds.labels
pos = labels.sum(axis=0)
neg = len(train_ds) - pos
pos_weight = torch.tensor(neg / (pos + 1e-6), dtype=torch.float32, device=device)

criterion = FocalLoss(alpha=pos_weight)
optimizer = optim.AdamW(
    model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
)

steps = len(train_loader)
scheduler = OneCycleLR(
    optimizer,
    max_lr=config.MAX_LR,
    steps_per_epoch=steps,
    epochs=config.EPOCHS,
    pct_start=0.3,
    anneal_strategy="cos",
)

history = {
    k: []
    for k in [
        "perte_entraînement",
        "perte_validation",
        "exactitude",
        "précision",
        "rappel",
        "score_f1",
    ]
}
best_val_loss = float("inf")

logging.info("🚀 Début de l'entraînement!")

for epoch in range(1, config.EPOCHS + 1):
    t0 = time.time()
    model.train()
    train_loss = 0

    for imgs, labels in tqdm(train_loader, desc=f"Train Ép {epoch}/{config.EPOCHS}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item() * imgs.size(0)

    train_loss /= len(train_ds)

    model.eval()
    val_loss, preds, vrais = 0, [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Val   Ép {epoch}/{config.EPOCHS}"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels.to(device))
            val_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds.append(probs)
            vrais.append(labels.numpy())

    val_loss /= len(val_loader.dataset)
    y_pred = np.vstack(preds) >= 0.5
    y_true = np.vstack(vrais) >= 0.5

    acc = accuracy_score(y_true.flatten(), y_pred.flatten())
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    history["perte_entraînement"].append(train_loss)
    history["perte_validation"].append(val_loss)
    history["exactitude"].append(acc)
    history["précision"].append(prec)
    history["rappel"].append(rec)
    history["score_f1"].append(f1)

    logging.info(
        f"Ép {epoch} | train {train_loss:.4f} | val {val_loss:.4f} | "
        f"Acc {acc:.4f} | Prec {prec:.4f} | Rec {rec:.4f} | F1 {f1:.4f} | "
        f"Durée {(time.time()-t0):.1f}s"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
        logging.info("🏆 Meilleur modèle sauvegardé")

# last epoch
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "last_model.pth"))
logging.info("✅ Modèle de la dernière époque sauvegardé")

# Plot metrics
epochs = range(1, config.EPOCHS + 1)
for k, v in history.items():
    plt.figure()
    plt.plot(epochs, v)
    plt.title(k)
    plt.xlabel("Ép.")
    plt.ylabel(k)
    plt.savefig(os.path.join(FIGURES_DIR, f"{k}.png"))
    plt.close()

logging.info("🎉 Entraînement terminé!")
