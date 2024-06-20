from src.stats.stats import readStats, updateModelPredicter
import torch

from src.predict.dataset import InterestDataset
from src.database.db import DB_URL

PREDICT_MODEL_PATH = "model/predict/model.pt"
EPOCHS = 40

class ArticlePredicterRNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(ArticlePredicterRNN, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.rnn = torch.nn.GRU(embedding_dim, hidden_dim, bidirectional=True, batch_first=True) # Random parameters as placeholders
        self.dropout = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        out = self.dropout(rnn_out[:, -1, :])
        out = self.fc(out)
        return torch.nn.functional.log_softmax(out, dim=1)

def train(model, dataloader):
    pass
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model.to(device)
    # model.to('cpu')
    
    # Criterion, optimizer and scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    num_epochs = 0
    loss = 0

    # Training the model
    while num_epochs < EPOCHS:
        for inputs, labels in dataloader:
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
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs, labels = inputs.to('cpu'), labels.to('cpu')

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            validation_loss += loss.item()


    scheduler.step()

    accuracy = correct / total
    avg_validation_loss = validation_loss / len(dataloader)

    print(f"\n{'-' * 20}\nFirst 200 articles\n\nTest accuracy: {accuracy * 100:.2f}%, with average validation loss: {avg_validation_loss:.6f}.")

    print(f"Finished testing.\n{'-' * 20}")
    with torch.no_grad():
        validation_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            validation_loss += loss.item()

    scheduler.step()

    accuracy = correct / total
    avg_validation_loss = validation_loss / len(dataloader)

    return accuracy, correct, total

def getData():
    pass

# Save model
def saveModel(model) -> bool:
    pass
    try:
        torch.save(model.state_dict(), PREDICT_MODEL_PATH)
        return True
    except Exception as e:
        print("Failed to save model.")
        print(e)
        return False

def loadModel(model) -> torch.nn.RNN:
    pass
    model.load_state_dict(torch.load(PREDICT_MODEL_PATH))
    model.eval()
    return model

def run():
    # Load data
    dataset = InterestDataset(DB_URL, 100, 300)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    # Prepare layers
    vocab_size = len(dataset.tokenizer.vocab)
    embedding_dim = 128 # 128 - 97.00%
    hidden_dim = 64 # 32 - 97.00%
    output_dim = len(set(dataset.labels))

    # Create RNN
    model = ArticlePredicterRNN(vocab_size, embedding_dim, hidden_dim, output_dim)

    # Train
    acc, correct, total = train(model, dataloader)
    print(f"Accuracy: {acc*100:.2f}% Correct/Total: {correct}/{total}")
    return model, acc, correct, total

if __name__ == "__main__":
    best_acc = 0
    best_correct = 0
    best_wrong = 0
    try:
        stats = readStats()
        predicter = stats["model"]["predicter"]
        best_acc = predicter["accuracy"]
        best_correct = predicter["predict_correct"]
        best_wrong = predicter["predict_wrong"]
    except Exception as e:
        print("[!] Failed to load stats")
    for n in range(5):
        model, acc, correct, total = run()
        if (acc > best_acc):
            print(f"Best accuracy so far {acc:.6f}. Saving...")
            if (saveModel(model)):
                print("Successfully saved predict model.")
            best_acc = acc
            best_correct = correct
            best_wrong = total - correct
    print(updateModelPredicter(best_acc, predicter["feedback_correct"], predicter["feedback_wrong"], best_correct, best_wrong))
    print(f"\n{'-'*20}END{'-'*20}\nBest accuracy ever for predict model {best_acc*100:.2f}%")
