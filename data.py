# data.py

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
test_transforms = val_transforms


class CheXpertDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.paths = df["Path"].values
        labels_df = (
            df[config.CLASSES]
            .replace(-1, 0)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )
        self.labels = labels_df.values.astype("float32")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.paths[idx]
        full = p if os.path.exists(p) else os.path.join(self.img_dir, p)
        img = Image.open(full).convert("RGB")
        if self.transform:
            img = self.transform(img)
        lbl = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, lbl


def get_dataloaders():
    df_train = pd.read_csv(config.TRAIN_CSV, low_memory=False)
    df_val = pd.read_csv(config.VAL_CSV, low_memory=False)
    df_test = pd.read_csv(config.TEST_CSV, low_memory=False)

    train_ds = CheXpertDataset(df_train, config.IMG_DIR, train_transforms)
    val_ds = CheXpertDataset(df_val, config.IMG_DIR, val_transforms)
    test_ds = CheXpertDataset(df_test, config.IMG_DIR, test_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    return train_loader, val_loader, test_loader, train_ds
