from src.stats.stats import readStats, updateModelPredicter
import torch

from src.predict.dataset import InterestDataset
from src.database.db import DB_URL, loadData, saveData

PREDICT_MODEL_PATH = "model/predict/model.pt"
EPOCHS = 25 # 50
LEARNING_RATE = 1e-3 # 0.1

# For debugging
torch.manual_seed(16)

class ArticlePredicterRNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(ArticlePredicterRNN, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.rnn = torch.nn.GRU(embedding_dim, hidden_dim, bidirectional=True, batch_first=True) # Random parameters as placeholders
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        out = self.dropout(rnn_out[:, -1, :])
        out = self.fc(out)
        return torch.nn.functional.log_softmax(out, dim=1)

def train(model, train_dataloader, val_dataloader):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model.to(device)
    
    # Criterion, optimizer and scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    num_epochs = 0
    loss = 0

    # Training the model
    while num_epochs < EPOCHS:
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs, labels = inputs.to('cpu'), labels.to('cpu')

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        num_epochs += 1
        print(f"Epoch: {num_epochs}, loss: {loss.item():.8f}")

    # Evaluation for training data
    model.eval()
    with torch.no_grad():
        validation_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs, labels = inputs.to('cpu'), labels.to('cpu')

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            validation_loss += loss.item()
        scheduler.step()

    print(f"Finished training.\n{'-' * 20}")

    return model, correct, total

def predictInterest():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"[i] Starting predicting interest...")

    dataset = InterestDataset("data/new_news.csv", 100)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)


    # Prepare layers
    vocab_size = len(dataset.tokenizer.vocab)
    embedding_dim = 128
    hidden_dim = 64
    output_dim = 10

    # TODO: Input size is equal to how many articles there are, which is bad bad
    # Create RNN
    model = ArticlePredicterRNN(vocab_size, embedding_dim, hidden_dim, output_dim)
    model = loadModel(model)
    model.to(device)

    df = loadData("data/new_news.csv")
    index = 0
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            for pred in predicted:
                predicted_label = int(pred)
                # print(f"Predicted label {predicted_label}")
                # Save new labels to dataframe
                df.at[index, "Interest_Rating"] = predicted_label
                index += 1
    # Save dataframe to test file
    saveData(df, "data/rated_news.csv")

    print(f"[i] Finished predicting interest.")

# Save model
def saveModel(model) -> bool:
    try:
        torch.save(model.state_dict(), PREDICT_MODEL_PATH)
        return True
    except Exception as e:
        print("Failed to save model.")
        print(e)
        return False

def loadModel(model) -> torch.nn.RNN:
    model.load_state_dict(torch.load(PREDICT_MODEL_PATH))
    model.eval()
    return model

def runTraining():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # Load data
    dataset = InterestDataset(DB_URL, 100)
    # Split data to train, validate and test datasets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Prepare layers
    vocab_size = len(dataset.tokenizer.vocab)
    embedding_dim = 128 # 128 - 97.00%
    hidden_dim = 64 # 32 - 97.00%
    # output_dim = len(set(dataset.labels))
    output_dim = 10

    # Create RNN
    model = ArticlePredicterRNN(vocab_size, embedding_dim, hidden_dim, output_dim)

    # Train
    model, correct, total = train(model, train_dataloader, val_dataloader)

    # Test
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct / total
    print(f"Test accuracy: {test_acc*100:.2f}% Correct/Total: {correct}/{total}")
    return model, test_acc, correct, total

def runPredicter():
    best_acc = 0
    best_correct = 0
    best_wrong = 0
    try:
        print(f"\n{'#'*20} Running Predicter {'#'*20}\n")
        stats = readStats()
        predicter = stats["model"]["predicter"]
        best_acc = predicter["accuracy"]
        best_correct = predicter["predict_correct"]
        best_wrong = predicter["predict_wrong"]
    except Exception as e:
        print("[!] Failed to load stats.\n{e}")
        return None
    # for n in range(5):
    #     print(f"\n{'#'*20} Running {n + 1} time out of 5 {'#'*20}\n")
    model, acc, correct, total = runTraining()
    if (acc > best_acc):
        print(f"Best accuracy so far {acc:.6f}. Saving...")
        if (saveModel(model)):
            print("Successfully saved predict model.")
        best_acc = acc
        best_correct = correct
        best_wrong = total - correct
    print(updateModelPredicter(round(best_acc, 4), predicter["feedback_correct"], predicter["feedback_wrong"], best_correct, best_wrong))
    print(f"\n{'-'*20}END{'-'*20}\nBest accuracy ever for predict model {best_acc*100:.2f}%")

if __name__ == "__main__":
    runPredicter()
    # predictInterest()
