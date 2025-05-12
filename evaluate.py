import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    multilabel_confusion_matrix
)

import config
from data import CheXpertDataset, val_transforms
from model import get_model

# 📂 Dossiers de sortie
OUTPUT_DIR = "outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# 🔧 Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "evaluate.log")),
        logging.StreamHandler(),
    ],
)

logging.info("⚙️  Lancement de l'évaluation du meilleur modèle")

# 🔍 Préparation du DataFrame de test
df_test = pd.read_csv(config.TEST_CSV, low_memory=False)
df_test["Path"] = df_test["Path"].astype(str).str.strip()
df_test = df_test[df_test["Path"] != "Path"]
df_test = df_test[
    df_test["Path"].apply(
        lambda p: os.path.exists(p) or os.path.exists(os.path.join(config.IMG_DIR, p))
    )
].reset_index(drop=True)
logging.info(f"✅ {len(df_test)} échantillons de test chargés")

# 📥 DataLoader de test
# CheXpertDataset renvoie (img, label, mask)
test_ds = CheXpertDataset(df_test, config.IMG_DIR, transform=val_transforms)
test_loader = DataLoader(
    test_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True
)
logging.info(f"ℹ️  DataLoader de test prêt ({len(test_loader)} batches)")

# 🏆 Chargement du meilleur modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)
model.load_state_dict(
    torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"), map_location=device)
)
model.eval()
logging.info("📦 Modèle chargé en mémoire")

# 🔮 Prédictions
all_preds, all_labels, all_masks = [], [], []
logging.info("🔮 Prédictions en cours...")
with torch.no_grad():
    for imgs, labels, mask in tqdm(test_loader, desc="Itération test"):
        imgs = imgs.to(device)
        out = model(imgs)
        probs = torch.sigmoid(out).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.numpy())
        all_masks.append(mask.numpy())
logging.info("✅ Prédictions terminées")

# Empiler résultats
probs = np.vstack(all_preds)
labels = np.vstack(all_labels)
masks = np.vstack(all_masks).astype(bool)

# 📊 Rapport par classe
logging.info("📊 Rapport de classification par pathologie (labels valides)")
report_lines = []
report_lines.append(f"{'Classe':30s} Précision  Rappel  F1-score  Support")
for idx, cls in enumerate(config.CLASSES):
    mask_j = masks[:, idx]
    if mask_j.sum() == 0:
        continue
    y_t = labels[:, idx][mask_j]
    y_p = (probs[:, idx][mask_j] >= 0.5)
    prec = precision_score(y_t, y_p, zero_division=0)
    rec  = recall_score(y_t, y_p, zero_division=0)
    f1   = f1_score(y_t, y_p, zero_division=0)
    sup  = int(mask_j.sum())
    report_lines.append(f"{cls:30s} {prec:>8.2f}   {rec:>5.2f}    {f1:>7.2f}   {sup:>7d}")
report = "\n".join(report_lines)
print(report)
logging.info("\n" + report)

# 📝 Metrics globales sur tous labels valides
y_true_flat = labels[masks]
y_pred_flat = (probs >= 0.5)[masks]
global_acc  = accuracy_score(y_true_flat, y_pred_flat)
macro_prec  = np.mean([precision_score(labels[:, i][masks[:, i]], (probs[:, i][masks[:, i]]>=0.5), zero_division=0)
                        for i in range(len(config.CLASSES))])
macro_rec   = np.mean([recall_score   (labels[:, i][masks[:, i]], (probs[:, i][masks[:, i]]>=0.5), zero_division=0)
                        for i in range(len(config.CLASSES))])
macro_f1    = np.mean([f1_score      (labels[:, i][masks[:, i]], (probs[:, i][masks[:, i]]>=0.5), zero_division=0)
                        for i in range(len(config.CLASSES))])
summary = (
    f"Global Acc: {global_acc:.4f} | "
    f"Prec: {macro_prec:.4f} | "
    f"Rec:  {macro_rec:.4f} | "
    f"F1:   {macro_f1:.4f}"
)
print(summary)
logging.info(summary)

# 🔢 Confusion matrix 2×2 par classe
logging.info("🔢 Generation des matrices de confusion 2×2 par classe")
from sklearn.metrics import confusion_matrix as cm_fn
# Calcul manuel par pathologie en respectant le mask
cms = []
for idx, cls in enumerate(config.CLASSES):
    mask_j = masks[:, idx]
    if mask_j.sum() == 0:
        cms.append(np.array([[0,0],[0,0]]))
        continue
    y_t = labels[:, idx][mask_j]
    y_p = (probs[:, idx][mask_j] >= 0.5)
    cm_i = cm_fn(y_t, y_p, labels=[0,1])  # [[TN, FP],[FN, TP]]
    cms.append(cm_i)

# Tracer un heatmap par classe
import seaborn as sns
fig, axes = plt.subplots(len(config.CLASSES), 1, figsize=(6, 3*len(config.CLASSES)))
for i, (cls, cm_i) in enumerate(zip(config.CLASSES, cms)):
    ax = axes[i]
    sns.heatmap(cm_i, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
    ax.set_xlabel('Pred')
    ax.set_ylabel('True')
    ax.set_title(f"{cls} [[TN, FP],[FN, TP]]")
plt.tight_layout()
conf2_path = os.path.join(FIGURES_DIR, "confusion_per_class.png")
plt.savefig(conf2_path)
plt.close()
logging.info(f"💾 Matrices de confusion par classe sauvegardées: {conf2_path}")
logging.info("🎉 Évaluation terminée !")
