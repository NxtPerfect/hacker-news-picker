import torch
from pandas import DataFrame
import matplotlib.pyplot as plt

# HACK: For debugging only
torch.manual_seed(16)

class Metrics():
    numOfEpochs = 0
    losses = []
    accuracies = []

def plotMetrics(model):
    plt.plot(model.metrics.losses)
    plt.xlabel("Num of epoch")
    plt.ylabel("Loss")
    plt.show()

def getDevice():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    return device

# Save model
def saveModel(model) -> bool:
    try:
        torch.save(model.state_dict(), model.model_path)
        return True
    except Exception as e:
        print("Failed to save model.")
        print(e)
        return False

def loadModel(model) -> torch.nn.RNN:
    model.load_state_dict(torch.load(model.model_path))
    model.eval()
    return model

def getTrainValidationTestDataloadersFromDataset(dataset):
    # Split data to train, validate and test datasets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


class PredicterRNN(torch.nn.Module):
    learning_rate = 1e-3 # 1e-3
    embedding_dim = 128
    hidden_dim = 64
    model_path = "model/predict/model.pt"
    predicted_database_path = "data/predicted_news.csv"
    device = "cuda"

    def __init__(self, vocab_size, output_dim):
        super(PredicterRNN, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, self.embedding_dim)
        self.rnn = torch.nn.GRU(self.embedding_dim, self.hidden_dim, bidirectional=True, batch_first=True) # Random parameters as placeholders
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(self.hidden_dim * 2, output_dim)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=25, gamma=0.1)

        self.device = getDevice()

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        out = self.dropout(rnn_out[:, -1, :])
        out = self.fc(out)
        return torch.nn.functional.log_softmax(out, dim=1)

class CategorizerRNN(torch.nn.Module):
    learning_rate = 1e-3 # 0.1 1e-2
    embedding_dim = 128
    hidden_dim = 128
    model_path = "model/categorize/model.pt"
    categorized_database_path = "data/categorized_news.csv"
    device = getDevice()
    metrics = Metrics()
    # device = "cpu"

    def __init__(self, vocab_size, output_dim=12):
        super(CategorizerRNN, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, self.embedding_dim)
        self.rnn = torch.nn.GRU(self.embedding_dim, self.hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = torch.nn.Dropout(0.8) # 0.5
        self.fc = torch.nn.Linear(self.hidden_dim * 2, output_dim)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)

        # self.device = getDevice()
        self.metrics = Metrics()

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        out = self.dropout(rnn_out[:, -1, :])
        out = self.fc(out)
        return torch.nn.functional.log_softmax(out, dim=1)

    def trainModel(self, EPOCHS, train_dataloader):
        num_epochs = 0
        loss = 0

        # Training the model
        while num_epochs < EPOCHS:
            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self(inputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(f"\rLoss: {loss:.8f}", end='')
            num_epochs += 1
            print(f"\nEpoch: {num_epochs} loss: {loss.item():.8f}")
            self.metrics.numOfEpochs = num_epochs
            self.metrics.losses.append(loss.item())

        return loss

    def evaluate(self, validation_dataloader, loss):
        # Evaluation for training data
        self.eval()
        with torch.no_grad():
            validation_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in validation_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # inputs, labels = inputs.to('cpu'), labels.to('cpu')

                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                validation_loss += loss.item()
            # Run scheduler at the end of each epoch
            self.scheduler.step(validation_loss)

        return correct, total
    def predict(self, df: DataFrame, dataloader, dataset):
        index = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)

                for pred in predicted:
                    predicted_label = dataset.category_labels[int(pred)]
                    # Save new labels to dataframe
                    df.at[index, "Category"] = predicted_label
                    index += 1
        return df
