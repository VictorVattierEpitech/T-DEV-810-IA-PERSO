# main.py

import os
import torch
from torchvision import transforms
from config import train_csv, valid_csv, images_dir, batch_size
from dataset import CheXpertDataset
from train import train_and_evaluate

def main():
    # 1) Device et répertoire de métriques
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    # 2) Transforms
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # 3) Chargement des datasets
    print(f"[main] Loading train dataset from {train_csv} (batch_size={batch_size})")
    train_ds = CheXpertDataset(
        csv_file=train_csv,
        images_dir=images_dir,
        transform=transform
    )
    print(f"[main]   → {len(train_ds)} samples loaded")

    print(f"[main] Loading validation dataset from {valid_csv}")
    val_ds = CheXpertDataset(
        csv_file=valid_csv,
        images_dir=images_dir,
        transform=transform
    )
    print(f"[main]   → {len(val_ds)} samples loaded\n")

    # 4) Entraînement + évaluation
    model = train_and_evaluate(
        train_ds=train_ds,
        val_ds=val_ds,
        device=device,
        metrics_dir=metrics_dir,
        use_sampler=True
    )

    # 5) Sauvegarde finale
    model_path = os.path.join(metrics_dir, "model_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining complete. Model saved to {model_path}")

if __name__ == "__main__":
    main()
