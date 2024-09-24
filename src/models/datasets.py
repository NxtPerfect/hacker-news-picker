from src.database.db import loadData
import torch
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder

class InterestDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, max_len, training=True) -> None:
        data = loadData(file_path)

        articles_count = min(data['Category'].isnull().idxmax(), data['Interest_Rating'].isnull().idxmax()) if training else 0

        if articles_count == 0:
            articles_count = len(data)
        data = data[:articles_count]

        self.max_len = max_len if max_len != -1 else articles_count
        self.features = (data["Title"] + ' ' + data["Category"]).values # also include categories
        self.labels = data["Interest_Rating"].values
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Cast labels to long integersunique_labels = sorted(set(self.labels))
        if articles_count != 0:
            unique_labels = sorted(set(self.labels))
            self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
            self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
            self.labels = torch.tensor([self.label_to_index[label] for label in self.labels], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.features[idx]
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        tokens = tokens[:self.max_len]

        padding_length = self.max_len - len(tokens)

        tokens = tokens + ([0] * padding_length)

        return torch.tensor(tokens), self.labels[idx]




class CategoryDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, training=True) -> None:
        data = loadData(file_path)

        # Find the first None value in the 'Title' column
        articles_count = data['Category'].isnull().idxmax() if training else 0
        
        # If no None values are found, use the entire dataset
        if articles_count == 0:
            articles_count = len(data)
        data = data[:articles_count]
        print(f"Picked: {articles_count} articles to train on.")

        self.max_len = articles_count
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
