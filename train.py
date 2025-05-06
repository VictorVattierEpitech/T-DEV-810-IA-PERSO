# train.py

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler
from config import pathology_labels, global_epochs, batch_size, learning_rate_global
from model import CheXpertModel

def compute_pos_weights(dataset, batch_size=256, device='cpu'):
    """
    Estime le pos_weight pour chaque classe : num_neg / num_pos.
    Affiche une barre de progression.
    """
    counts = np.zeros(len(pathology_labels), dtype=np.float64)
    total = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    for _, labels in tqdm(loader, desc="Computing pos_weights"):
        batch = torch.stack([labels[c] for c in pathology_labels], dim=1).numpy()
        counts += batch.sum(axis=0)
        total += batch.shape[0]
    neg = total - counts
    pos_weights = neg / (counts + 1e-6)
    return torch.tensor(pos_weights, dtype=torch.float32).to(device)

def save_plot(x, ys, labels, title, xlabel, ylabel, save_path):
    """
    Trace et sauvegarde un graphique de plusieurs courbes.
      x : liste d'abscisses
      ys : liste de listes (chaque liste de même longueur que x)
      labels : noms des courbes
    """
    plt.figure()
    for y_vals, lab in zip(ys, labels):
        plt.plot(x, y_vals, label=lab)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.savefig(save_path)
    plt.close()

def train_and_evaluate(train_ds, val_ds, device, metrics_dir="metrics", use_sampler=True):
    """
    Entraîne et évalue le modèle. Trace loss, AUC et matrices de confusion après chaque epoch.
    Sauvegarde 'last.pt' et 'best.pt' dans metrics_dir.
    """
    os.makedirs(metrics_dir, exist_ok=True)

    # 1) Pos_weights et criterion
    pos_weights = compute_pos_weights(train_ds, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # 2) DataLoaders
    if use_sampler:
        pw = pos_weights.cpu().numpy()
        weights = []
        for _, labels in tqdm(train_ds, desc="Computing sample weights"):
            lab = np.array([labels[c] for c in pathology_labels], dtype=np.float32)
            w = (lab * pw).sum() / (lab.sum() + 1e-6)
            weights.append(w + 1.0)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # 3) Modèle, optimiseur, scheduler
    model     = CheXpertModel(dropout_prob=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_global)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 4) Historique + best checkpoint
    train_losses, val_losses = [], []
    auc_history = {label: [] for label in pathology_labels}
    best_val_loss = float('inf')

    # 5) Boucle d'entraînement
    for epoch in range(1, global_epochs + 1):
        # --- Train ---
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch}/{global_epochs}"):
            images = images.to(device)
            y_true = torch.stack([labels[c] for c in pathology_labels], dim=1).float().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, dict):
                logits = torch.stack([outputs[c].squeeze(-1) for c in pathology_labels], dim=1)
            else:
                logits = outputs

            loss = criterion(logits, y_true)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # --- Validation ---
        model.eval()
        running_val_loss = 0.0
        all_y_true, all_y_prob = [], []
        for images, labels in tqdm(val_loader, desc=f"Val Epoch {epoch}/{global_epochs}"):
            images = images.to(device)
            y_true = torch.stack([labels[c] for c in pathology_labels], dim=1).float().to(device)
            with torch.no_grad():
                outputs = model(images)
                if isinstance(outputs, dict):
                    logits = torch.stack([outputs[c].squeeze(-1) for c in pathology_labels], dim=1)
                else:
                    logits = outputs

            loss = criterion(logits, y_true)
            running_val_loss += loss.item()
            all_y_true.append(y_true.cpu().numpy())
            all_y_prob.append(torch.sigmoid(logits).cpu().numpy())

        val_loss = running_val_loss / len(val_loader)
        val_losses.append(val_loss)

        # --- Checkpoints ---
        torch.save(model.state_dict(), os.path.join(metrics_dir, 'last.pt'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(metrics_dir, 'best.pt'))

        # --- Metrics & plots ---
        y_true_arr = np.concatenate(all_y_true, axis=0)
        y_prob_arr = np.concatenate(all_y_prob, axis=0)
        y_pred_arr = (y_prob_arr >= 0.5).astype(int)

        # AUC
        aucs = roc_auc_score(y_true_arr, y_prob_arr, average=None)
        for idx, label in enumerate(pathology_labels):
            auc_history[label].append(aucs[idx])
        save_plot(
            x=list(range(1, epoch + 1)),
            ys=[auc_history[label] for label in pathology_labels],
            labels=pathology_labels,
            title="AUC par classe",
            xlabel="Epoch",
            ylabel="AUC",
            save_path=os.path.join(metrics_dir, f"auc_curve_epoch_{epoch}.png")
        )

        # Confusion matrices
        # --- Matrice de confusion agrégée sur toutes les classes ---
        from sklearn.metrics import confusion_matrix

        # Aplatir tous les labels (« multilabel » → binaire global)
        y_true_flat = y_true_arr.ravel()
        y_pred_flat = y_pred_arr.ravel()

        cm_all = confusion_matrix(y_true_flat, y_pred_flat)
        plt.figure(figsize=(6, 6))
        plt.imshow(cm_all, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"Confusion matrix (toutes classes) - Epoch {epoch}")
        plt.colorbar()

        tick_labels = ['Négatif', 'Positif']
        plt.xticks([0, 1], tick_labels)
        plt.yticks([0, 1], tick_labels)
        plt.xlabel("Prédit")
        plt.ylabel("Vrai")

        # Annoter les cellules
        for i in range(cm_all.shape[0]):
            for j in range(cm_all.shape[1]):
                plt.text(j, i, format(cm_all[i, j], 'd'),
                        ha='center', va='center')

        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, f"conf_all_classes_epoch_{epoch}.png"))
        plt.close()


        # Loss curves
        save_plot(
            x=list(range(1, epoch + 1)),
            ys=[train_losses, val_losses],
            labels=["Train Loss", "Val Loss"],
            title="Loss",
            xlabel="Epoch",
            ylabel="Loss",
            save_path=os.path.join(metrics_dir, f"loss_curve_epoch_{epoch}.png")
        )

        print(f"Epoch {epoch}/{global_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

    return model
