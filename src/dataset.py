"""Load images from an explicit, QC-approved manifest."""
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):
    def __init__(self, manifest_path, image_dir, transform=None):
        self.table = pd.read_csv(manifest_path, dtype={'Patient ID':str})
        required = {'Image Index', 'Patient ID', 'pneumonia'}
        if not required.issubset(self.table.columns):
            raise ValueError('Manifest lacks required columns.')
        if self.table['Image Index'].duplicated().any():
            raise ValueError('Manifest contains duplicate filenames.')
        if not self.table['pneumonia'].isin([0,1]).all():
            raise ValueError('Targets must be 0 or 1.')
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        with Image.open(self.image_dir / row['Image Index']) as source:
            image = source.copy()
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(float(row['pneumonia']), dtype=torch.float32)
