import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
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

# 📥 Construction du DataLoader de test
test_ds = CheXpertDataset(df_test, config.IMG_DIR, transform=val_transforms)
test_loader = DataLoader(
    test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS
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

# 🔢 Prédictions
all_preds, all_labels = [], []
logging.info("🔮 Prédictions en cours...")
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Itération test"):
        imgs = imgs.to(device)
        out = model(imgs)
        probs = torch.sigmoid(out).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.numpy())
logging.info("✅ Prédictions terminées")

probs = np.vstack(all_preds)
y_pred = probs >= 0.5
y_true = np.vstack(all_labels) >= 0.5

# 📝 Rapport de classification
logging.info("📊 Rapport de classification")
report = classification_report(
    y_true, y_pred, target_names=config.CLASSES, zero_division=0
)
print(report)
logging.info("\n" + report)
summary = (
    f"Global Acc: {accuracy_score(y_true.flatten(), y_pred.flatten()):.4f} | "
    f"Prec: {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f} | "
    f"Rec:  {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f} | "
    f"F1:   {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
)
print(summary)
logging.info(summary)

# 📊 Matrice de confusion 14×14 annotée
logging.info("🔢 Génération de la matrice de confusion")
y_true_mc = np.argmax(y_true.astype(int), axis=1)
y_pred_mc = np.argmax(probs, axis=1)
cm = confusion_matrix(y_true_mc, y_pred_mc, labels=list(range(len(config.CLASSES))))

plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation="nearest", aspect="auto", cmap="Blues")
plt.title("Matrice de confusion 14×14")
plt.colorbar()
ticks = np.arange(len(config.CLASSES))
plt.xticks(ticks, config.CLASSES, rotation=90)
plt.yticks(ticks, config.CLASSES)
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )
plt.tight_layout()
conf_path = os.path.join(FIGURES_DIR, "matrice_confusion_14x14.png")
plt.savefig(conf_path)
plt.close()
logging.info(f"💾 Matrice de confusion sauvegardée: {conf_path}")

# 📈 Courbes ROC & AUC sur un seul plot
logging.info("📈 Génération des courbes ROC & calcul AUC")
plt.figure(figsize=(10, 8))
for idx, cls in enumerate(config.CLASSES):
    try:
        auc = roc_auc_score(y_true[:, idx].astype(int), probs[:, idx])
    except ValueError:
        auc = float("nan")
    fpr, tpr, _ = roc_curve(y_true[:, idx].astype(int), probs[:, idx])
    plt.plot(fpr, tpr, label=f"{cls} (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Aléatoire")
plt.title("Courbes ROC multi-classes")
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.legend(loc="lower right", fontsize="small")
plt.grid(True)
plt.tight_layout()
roc_path = os.path.join(FIGURES_DIR, "roc_auc_global.png")
plt.savefig(roc_path)
plt.close()
logging.info(f"💾 Courbes ROC sauvegardées: {roc_path}")

logging.info("🎉 Évaluation terminée !")
