import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from src.database.db import DB_URL


class CategoryDataset(Dataset):
    def __init__(self) -> None:
        data = pd.read_csv(DB_URL)
        features = data["Title"].values
        labels = data["Category"].values

        # TfidfVectorizer convert text to numbers and to tensor
        enc = TfidfVectorizer()
        self.features = torch.from_numpy(enc.fit_transform(features).toarray())

        # LabelEncoding to numerical values
        enc = LabelEncoder()
        self.labels = torch.from_numpy(enc.fit_transform(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
