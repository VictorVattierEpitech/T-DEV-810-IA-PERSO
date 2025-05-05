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
    """
    counts = np.zeros(len(pathology_labels), dtype=np.float64)
    total = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    for _, labels in loader:
        batch = torch.stack([labels[c] for c in pathology_labels], dim=1).numpy()
        counts += batch.sum(axis=0)
        total += batch.shape[0]
    neg = total - counts
    pos_weights = neg / (counts + 1e-6)
    return torch.tensor(pos_weights, dtype=torch.float32).to(device)


def save_plot(x, ys, labels, title, xlabel, ylabel, save_path):
    """Trace et sauvegarde un graphique"""
    plt.figure()
    for y, label in zip(ys, labels):
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def train_and_evaluate(train_ds, val_ds, device, metrics_dir="metrics", use_sampler=True):
    """
    Entraîne et évalue le modèle, trace loss, AUC et matrices de confusion
    à chaque époque et sauvegarde les graphiques dans metrics_dir.
    """
    os.makedirs(metrics_dir, exist_ok=True)

    # 1) Loss pondérée
    pos_weights = compute_pos_weights(train_ds, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # 2) DataLoaders
    if use_sampler:
        pw = pos_weights.cpu().numpy()
        weights = []
        for _, labels in train_ds:
            lab = np.array([labels[c] for c in pathology_labels], dtype=np.float32)
            w = (lab * pw).sum() / (lab.sum() + 1e-6)
            weights.append(w + 1.0)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # 3) Modèle, optimiseur, scheduler
    model = CheXpertModel(dropout_prob=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_global)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # 4) Boucle d'entraînement
    train_losses, val_losses = [], []
    for epoch in range(1, global_epochs + 1):
        # Entraînement
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch}/{global_epochs}"):
            images = images.to(device)
            y_true = torch.stack([labels[c] for c in pathology_labels], dim=1).float().to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, y_true)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        all_y_true, all_y_prob = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                y_true = torch.stack([labels[c] for c in pathology_labels], dim=1).float().to(device)
                logits = model(images)
                loss = criterion(logits, y_true)
                running_val_loss += loss.item()
                all_y_true.append(y_true.cpu().numpy())
                all_y_prob.append(torch.sigmoid(logits).cpu().numpy())
        val_loss = running_val_loss / len(val_loader)
        val_losses.append(val_loss)

        # Concaténation
        y_true_arr = np.concatenate(all_y_true, axis=0)
        y_prob_arr = np.concatenate(all_y_prob, axis=0)
        y_pred_arr = (y_prob_arr >= 0.5).astype(int)

        # AUC par classe
        aucs = roc_auc_score(y_true_arr, y_prob_arr, average=None)
        save_plot(
            x=list(range(1, epoch + 1)),
            ys=[aucs],
            labels=pathology_labels,
            title=f"Epoch {epoch} AUC", xlabel="Epoch", ylabel="AUC",
            save_path=os.path.join(metrics_dir, f"epoch_{epoch}_auc.png")
        )

        # Matrices de confusion
        for idx, label in enumerate(pathology_labels):
            cm = confusion_matrix(y_true_arr[:, idx], y_pred_arr[:, idx])
            plt.figure()
            plt.imshow(cm, interpolation='nearest')
            plt.title(f"Confusion Matrix {label} - Epoch {epoch}")
            plt.colorbar()
            plt.xticks([0, 1]); plt.yticks([0, 1])
            plt.xlabel("Predicted"); plt.ylabel("True")
            plt.savefig(os.path.join(metrics_dir, f"epoch_{epoch}_conf_{label}.png"))
            plt.close()

        # Courbes de loss
        save_plot(
            x=list(range(1, epoch + 1)),
            ys=[train_losses, val_losses],
            labels=["Train Loss", "Val Loss"],
            title="Loss Curve", xlabel="Epoch", ylabel="Loss",
            save_path=os.path.join(metrics_dir, f"epoch_{epoch}_loss.png")
        )

        print(f"Epoch {epoch}/{global_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model
