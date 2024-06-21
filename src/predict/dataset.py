from src.database.db import loadData
import torch
from transformers import BertTokenizer


class InterestDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, max_len=100, articles_count=200) -> None:
        data = loadData(file_path)
        if articles_count != 0:
            data = data[:articles_count]
        self.max_len = max_len
        self.features = (data["Title"] + ' ' + data["Category"]).values # also include categories
        self.labels = data["Interest_Rating"].values
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Cast labels to long integers
        self.labels = torch.tensor(self.labels, dtype=torch.long)

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
