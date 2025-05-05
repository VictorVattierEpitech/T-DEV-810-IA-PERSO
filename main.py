# main.py

import os
import torch
from torchvision import transforms
from config import train_csv, valid_csv, images_dir, global_epochs, batch_size, learning_rate_global
from train import train_and_evaluate
from your_dataset import CheXpertDataset  # Adaptez selon votre implémentation


def main():
    """
    Script principal : prépare les datasets, lance l'entraînement et sauvegarde le modèle.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    train_ds = CheXpertDataset(
        csv_file=train_csv,
        images_dir=images_dir,
        split="train",
        transform=transform
    )
    val_ds = CheXpertDataset(
        csv_file=valid_csv,
        images_dir=images_dir,
        split="valid",
        transform=transform
    )

    model = train_and_evaluate(
        train_ds=train_ds,
        val_ds=val_ds,
        device=device,
        metrics_dir=metrics_dir,
        use_sampler=True
    )

    torch.save(
        model.state_dict(),
        os.path.join(metrics_dir, "model_final.pth")
    )
    print("Entraînement terminé. Modèle et métriques disponibles dans le dossier 'metrics'.")


if __name__ == "__main__":
    main()
