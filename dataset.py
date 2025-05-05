# dataset.py

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from config import pathology_labels

class CheXpertDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform
        self.label_columns = pathology_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["Path"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        labels = {}
        for label in self.label_columns:
            value = row[label]
            try:
                label_value = float(value) if pd.notna(value) and value != "" else 0.0
                if label_value not in [0.0, 1.0]:
                    label_value = 0.0
                labels[label] = label_value
            except Exception:
                labels[label] = 0.0

        return image, labels
