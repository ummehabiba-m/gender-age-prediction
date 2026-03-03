## dataset.py

from torch.utils.data import Dataset
from PIL import Image
import pandas as pd, os, torch

class FairFaceDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['file'])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        gender = torch.tensor(1.0 if row['gender'] == 'Male' else 0.0)

        # Handle age range like "10-19"
        age_str = str(row['age'])
        if '-' in age_str:
            low, high = age_str.split('-')
            age_val = (float(low) + float(high)) / 2
        else:
            age_val = float(age_str)
        age = torch.tensor(age_val)

        return img, gender, age