import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config

# ici on prépare les transformations à appliquer sur les images

# pour l'entraînement on redimensionne l'image
# on ajoute une symétrie horizontale aléatoire pour augmenter les données
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

# pour la validation on ne fait que redimensionner et normaliser
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

# on utilise les mêmes transformations pour le test que pour la validation
test_transforms = val_transforms

# on définit ici notre dataset personnalisé basé sur un fichier csv
class CheXpertDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)  # on réinitialise les index du dataframe
        self.img_dir = img_dir               # dossier contenant les images
        self.transform = transform           # transformations à appliquer

        # on récupère les colonnes des labels et on remplace les valeurs manquantes par -1
        labels_df = df[config.CLASSES].apply(pd.to_numeric, errors='coerce').fillna(-1)
        orig = labels_df.values.astype(int)

        # on fait une copie pour gérer les valeurs incertaines selon une stratégie définie
        mapped = orig.astype('float32').copy()
        for j, cls in enumerate(config.CLASSES):
            strat = config.UNCERTAIN_LABEL_MAPPING.get(cls, 'ignore')
            if strat == 'zeros':
                mapped[:, j] = np.where(orig[:, j] == -1, 0, orig[:, j])
            elif strat == 'ones':
                mapped[:, j] = np.where(orig[:, j] == -1, 1, orig[:, j])
            # si ignore on ne change rien

        # on crée un masque pour savoir quels labels sont valides
        mask = (mapped != -1).astype('float32')

        # on remplace les -1 par 0 juste pour que le modèle puisse calculer une perte
        mapped = np.where(mapped == -1, 0, mapped)

        # on convertit tout ça en tenseurs pytorch
        self.labels = torch.from_numpy(mapped)
        self.mask   = torch.from_numpy(mask)
        self.paths  = df['Path'].values

    def __len__(self):
        # retourne le nombre d'images dans le dataset
        return len(self.df)

    def __getitem__(self, idx):
        # ici on charge l'image à l'index idx
        p = self.paths[idx]
        full = p if os.path.exists(p) else os.path.join(self.img_dir, p)
        img = Image.open(full).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], self.mask[idx]

# fonction pour créer les data loaders pour entraînement validation et test
def get_dataloaders():
    # on lit les fichiers csv qui contiennent les annotations
    df_train = pd.read_csv(config.TRAIN_CSV, low_memory=False)
    df_val   = pd.read_csv(config.VAL_CSV,   low_memory=False)
    df_test  = pd.read_csv(config.TEST_CSV,  low_memory=False)

    # on crée les datasets pour chaque split
    train_ds = CheXpertDataset(df_train, config.IMG_DIR, train_transforms)
    val_ds   = CheXpertDataset(df_val,   config.IMG_DIR, val_transforms)
    test_ds  = CheXpertDataset(df_test,  config.IMG_DIR, test_transforms)

    # on crée les data loaders qui servent à charger les données en batchs
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,               # on mélange les données pour l'entraînement
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,              # on ne mélange pas pour la validation
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,              # idem pour le test
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_ds
