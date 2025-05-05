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
    Parcourt le dataset pour calculer pos_weight = #neg / #pos par classe.
    """
    counts = np.zeros(len(pathology_labels), dtype=np.float64)
    total = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for _, labels in tqdm(loader, desc="→ pos_weights", leave=True):
        batch = torch.stack([labels[c] for c in pathology_labels], dim=1).numpy()
        counts += batch.sum(axis=0)
        total += batch.shape[0]
    neg = total - counts
    pos_weights = neg / (counts + 1e-6)
    return torch.tensor(pos_weights, dtype=torch.float32).to(device)


def save_plot(x, ys, labels, title, xlabel, ylabel, save_path):
    """Trace un graphique et le sauve sur disque."""
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
    os.makedirs(metrics_dir, exist_ok=True)
    print(f"[train] Start training: {global_epochs} epochs", flush=True)

    # 1) Loss pondérée
    pos_weights = compute_pos_weights(train_ds, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # 2) DataLoader avec sampler ou shuffle
    if use_sampler:
        pw = pos_weights.cpu().numpy()
        weights = []
        for _, labels in train_ds:
            lab = np.array([labels[c] for c in pathology_labels], dtype=np.float32)
            w = (lab * pw).sum() / (lab.sum() + 1e-6)
            weights.append(w + 1.0)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  sampler=sampler, num_workers=0)
        print("[train] Using WeightedRandomSampler", flush=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    # 3) Modèle, optimiseur, scheduler
    model = CheXpertModel(dropout_prob=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_global)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # 4) Boucle d'entraînement
    train_losses, val_losses = [], []
    for epoch in range(1, global_epochs + 1):
        print(f"\n=== Epoch {epoch}/{global_epochs} ===", flush=True)

        # --- Entraînement ---
        model.train()
        running_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Train  {epoch}/{global_epochs}", leave=True)
        for i, (images, labels) in enumerate(train_loop, 1):
            images = images.to(device)
            y_true = torch.stack([labels[c] for c in pathology_labels], dim=1).float().to(device)

            optimizer.zero_grad()
            raw = model(images)
            # Concatène si le modèle renvoie un dict label→(B,1)
            if isinstance(raw, dict):
                logits = torch.cat([raw[c] for c in pathology_labels], dim=1)
            else:
                logits = raw
            loss = criterion(logits, y_true)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_loop.set_postfix(loss=f"{running_loss/i:.4f}")

        train_losses.append(running_loss / len(train_loader))

        # --- Validation ---
        model.eval()
        running_val = 0.0
        all_true, all_prob = [], []
        val_loop = tqdm(val_loader, desc=f"Valid  {epoch}/{global_epochs}", leave=True)
        for j, (images, labels) in enumerate(val_loop, 1):
            images = images.to(device)
            y_true = torch.stack([labels[c] for c in pathology_labels], dim=1).float().to(device)

            raw = model(images)
            if isinstance(raw, dict):
                logits = torch.cat([raw[c] for c in pathology_labels], dim=1)
            else:
                logits = raw
            loss = criterion(logits, y_true)

            running_val += loss.item()
            val_loop.set_postfix(val_loss=f"{running_val/j:.4f}")

            all_true.append(y_true.cpu().numpy())
            all_prob.append(torch.sigmoid(logits).cpu().numpy())

        val_losses.append(running_val / len(val_loader))

        # --- Metrics ---
        y_true_arr = np.concatenate(all_true, axis=0)
        y_prob_arr = np.concatenate(all_prob, axis=0)
        y_pred_arr = (y_prob_arr >= 0.5).astype(int)

        # AUC
        aucs = roc_auc_score(y_true_arr, y_prob_arr, average=None)
        save_plot(
            x=list(range(1, epoch+1)), ys=[aucs], labels=pathology_labels,
            title=f"Epoch {epoch} AUC", xlabel="Epoch", ylabel="AUC",
            save_path=os.path.join(metrics_dir, f"epoch_{epoch}_auc.png")
        )

        # Matrices de confusion
        for idx, label in enumerate(pathology_labels):
            cm = confusion_matrix(y_true_arr[:, idx], y_pred_arr[:, idx])
            plt.figure()
            plt.imshow(cm, interpolation='nearest')
            plt.title(f"CM {label} - Epoch {epoch}")
            plt.colorbar()
            plt.xticks([0,1]); plt.yticks([0,1])
            plt.xlabel("Predicted"); plt.ylabel("True")
            plt.savefig(os.path.join(metrics_dir, f"epoch_{epoch}_cm_{label}.png"))
            plt.close()

        # Courbe de loss
        save_plot(
            x=list(range(1, epoch+1)),
            ys=[train_losses, val_losses],
            labels=["Train Loss", "Val Loss"],
            title="Loss Curve", xlabel="Epoch", ylabel="Loss",
            save_path=os.path.join(metrics_dir, f"epoch_{epoch}_loss.png")
        )

        print(f"[Epoch {epoch}] Train: {train_losses[-1]:.4f} | Val: {val_losses[-1]:.4f}", flush=True)
        scheduler.step(val_losses[-1])

    return model
