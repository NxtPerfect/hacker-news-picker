import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer


class CategoryDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, max_len=100, articles_count=200) -> None:
        data = pd.read_csv(file_path)
        if articles_count != 0:
            data = data[:articles_count]
        self.max_len = max_len
        self.features = data["Title"].values
        self.labels = data["Category"].values
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # # LabelEncoding to numerical values
        self.label_encoder = LabelEncoder()
        self.labels = torch.tensor(self.label_encoder.fit_transform(self.labels))

        # Nonencoded labels
        self.category_labels = {index: label for index, label in enumerate(self.label_encoder.classes_)}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.features[idx]
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        tokens = tokens[:self.max_len]

        padding_length = self.max_len - len(tokens)

        tokens = tokens + ([0] * padding_length)

        return torch.tensor(tokens), self.labels[idx]
        # return self.features[idx], self.labels[idx]
