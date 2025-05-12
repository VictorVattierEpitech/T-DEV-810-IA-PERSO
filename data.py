import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config

# Transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])
test_transforms = val_transforms

class CheXpertDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        # Lire les labels et remplacer NaN par -1
        labels_df = df[config.CLASSES].apply(pd.to_numeric, errors='coerce').fillna(-1)
        orig = labels_df.values.astype(int)

        # Mapping incertains
        mapped = orig.astype('float32').copy()
        for j, cls in enumerate(config.CLASSES):
            strat = config.UNCERTAIN_LABEL_MAPPING.get(cls, 'ignore')
            if strat == 'zeros':
                mapped[:, j] = np.where(orig[:, j] == -1, 0, orig[:, j])
            elif strat == 'ones':
                mapped[:, j] = np.where(orig[:, j] == -1, 1, orig[:, j])
            # ignore: leave -1
        # Mask des labels valides
        mask = (mapped != -1).astype('float32')
        # Remplacer -1 par 0 pour l'entrée de la loss
        mapped = np.where(mapped == -1, 0, mapped)

        self.labels = torch.from_numpy(mapped)
        self.mask   = torch.from_numpy(mask)
        self.paths  = df['Path'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.paths[idx]
        full = p if os.path.exists(p) else os.path.join(self.img_dir, p)
        img = Image.open(full).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], self.mask[idx]


def get_dataloaders():
    df_train = pd.read_csv(config.TRAIN_CSV, low_memory=False)
    df_val   = pd.read_csv(config.VAL_CSV,   low_memory=False)
    df_test  = pd.read_csv(config.TEST_CSV,  low_memory=False)

    train_ds = CheXpertDataset(df_train, config.IMG_DIR, train_transforms)
    val_ds   = CheXpertDataset(df_val,   config.IMG_DIR, val_transforms)
    test_ds  = CheXpertDataset(df_test,  config.IMG_DIR, test_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_ds