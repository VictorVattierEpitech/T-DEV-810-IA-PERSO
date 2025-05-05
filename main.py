# main.py

import os
import torch
from torchvision import transforms
from config import train_csv, valid_csv, images_dir, global_epochs, batch_size
from dataset import CheXpertDataset
from train import train_and_evaluate

def main():
    # Device & dossier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Chargement des datasets
    print(f"[main] Chargement train: {train_csv} → {batch_size} samples/batch")
    train_ds = CheXpertDataset(csv_file=train_csv,
                               images_dir=images_dir,
                               transform=transform)
    print(f"[main]   → {len(train_ds)} images")
    val_ds = CheXpertDataset(csv_file=valid_csv,
                             images_dir=images_dir,
                             transform=transform)
    print(f"[main] Validation: {valid_csv} → {len(val_ds)} images\n")

    # Entraînement + métriques
    model = train_and_evaluate(train_ds=train_ds,
                               val_ds=val_ds,
                               device=device,
                               metrics_dir=metrics_dir,
                               use_sampler=True)

    # Sauvegarde finale
    torch.save(model.state_dict(),
               os.path.join(metrics_dir, "model_final.pth"))
    print("\n→ Entraînement terminé, résultats dans 'metrics/'")

if __name__ == "__main__":
    main()
