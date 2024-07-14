from src.stats.stats import readStats, updateModelCategorizer
import torch

from src.categorize.dataset import CategoryDataset
from src.database.db import DB_URL, loadData, saveData

CATEGORIZE_MODEL_PATH = "model/categorize/model.pt"
EPOCHS = 100
ARTICLES_COUNT = 1800
LEARNING_RATE = 1e-3 # 0.1
EMBEDDING_DIM = 128
HIDDEN_DIM = 64

class ArticleCategorizerRNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, output_dim=12):
        super(ArticleCategorizerRNN, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.rnn = torch.nn.GRU(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        out = self.dropout(rnn_out[:, -1, :])
        out = self.fc(out)
        return torch.nn.functional.log_softmax(out, dim=1)

def train(device, model, train_dataloader, validation_dataloader):
    model.to(device)
    
    # Criterion, optimizer and scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # 0.1
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1) # 25 0.1
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    num_epochs = 0
    loss = 0

    # Training the model
    while num_epochs < EPOCHS:
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

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
        for inputs, labels in validation_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs, labels = inputs.to('cpu'), labels.to('cpu')

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            validation_loss += loss.item()
        # Run scheduler at the end of each epoch
        scheduler.step(validation_loss)

    print(f"\n{'-' * 20}\nFirst {ARTICLES_COUNT} articles\n\n")
    print(f"Finished training.\n{'-' * 20}")

    return model, correct, total

def predictCategory():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"[i] Started predicting category...")

    dataset = CategoryDataset(DB_URL, 100, 0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)


    # Prepare layers
    vocab_size = len(dataset.tokenizer.vocab)
    embedding_dim = EMBEDDING_DIM
    hidden_dim = HIDDEN_DIM
    output_dim = len(set(dataset.labels.numpy())) - 1 # Exclude null values

    # Create RNN
    model = ArticleCategorizerRNN(vocab_size, embedding_dim, hidden_dim, output_dim)
    model = loadModel(model)
    model.to(device)

    df = loadData()
    index = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            for pred in predicted:
                predicted_label = dataset.category_labels[int(pred)]
                # print(f"Predicted label {predicted_label}")
                # Save new labels to dataframe
                df.at[index, "Category"] = predicted_label
                index += 1
    # Save dataframe to test file
    saveData(df, "data/categorized_news.csv")

    print(f"[i] Finished predicting category.")

# Save model
def saveModel(model) -> bool:
    try:
        torch.save(model.state_dict(), CATEGORIZE_MODEL_PATH)
        return True
    except Exception as e:
        print("Failed to save model.")
        print(e)
        return False

def loadModel(model) -> torch.nn.RNN:
    model.load_state_dict(torch.load(CATEGORIZE_MODEL_PATH))
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
    dataset = CategoryDataset(DB_URL, 100, ARTICLES_COUNT)

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
    embedding_dim = EMBEDDING_DIM # 128 - 98.67% // 256 - 38%
    hidden_dim = HIDDEN_DIM # 32 - 98.67% // 64 - 38%
    output_dim = len(set(dataset.labels.numpy()))

    # Create RNN
    model = ArticleCategorizerRNN(vocab_size, embedding_dim, hidden_dim, output_dim)

    # Train
    model, correct, total = train(device, model, train_dataloader, val_dataloader)

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

def findParams():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load data
    dataset = CategoryDataset(DB_URL, 100, ARTICLES_COUNT)

    # Split data to train, validate and test datasets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    learning_rate = [0.1, 0.01, 0.001, 0.0001]
    embed_sizes = [64, 128, 256, 512, 1024]
    hidden_dims = [32, 64, 128, 256]
    epochs = [10, 20, 30, 40, 50, 100]

    correct = 0
    total = 0
    test_acc = 0
    best_acc = -1

    for lr in learning_rate:
        for es in embed_sizes:
            for epoch in epochs:
                for hd in hidden_dims:
                    # Prepare layers
                    vocab_size = len(dataset.tokenizer.vocab)
                    embedding_dim = es # 128 - 98.67% // 256 - 38%
                    hidden_dim = hd # 32 - 98.67% // 64 - 38%
                    output_dim = len(set(dataset.labels.numpy()))

                    # Create RNN
                    model = ArticleCategorizerRNN(vocab_size, embedding_dim, hidden_dim, output_dim)
                    # Train
                    model, correct, total = train(device, model, train_dataloader, val_dataloader, lr, epoch)

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
                    if test_acc > best_acc:
                        print(f"Best accuracy {best_acc*100:.2f}% with learning rate: {lr} embed_size: {es} epochs: {epoch} hidden_dim: {hd}")
                        # 0.1 64 20 64 27%
                        # 0.1 128 100 32 28.57%
                        # 0.1 256 10 256 32.86%
                        # 0.1 512 40 256 37.14%
                        best_acc = test_acc
    return model, test_acc, correct, total

def runCategorizer():
    best_acc = 0
    best_correct = 0
    best_wrong = 0
    try:
        print(f"\n{'#'*20} Running Categorizer {'#'*20}\n")
        stats = readStats()
        categorizer = stats["model"]["categorizer"]
        best_acc = categorizer["accuracy"]
        best_correct = categorizer["predict_correct"]
        best_wrong = categorizer["predict_wrong"]
    except Exception as e:
        print("[!] Failed to load stats.\n{e}")
        return None
    for n in range(5):
        print(f"\n{'#'*20} Running {n + 1} time out of 5 {'#'*20}\n")
        model, acc, correct, total = runTraining()
        if (acc > best_acc):
            print(f"Best accuracy so far {acc:.6f}. Saving...")
            if (saveModel(model)):
                print("Successfully saved categorizer model.")
            best_acc = acc
            best_correct = correct
            best_wrong = total - correct
    print(updateModelCategorizer(round(best_acc, 4), categorizer["feedback_correct"], categorizer["feedback_wrong"], best_correct, best_wrong))
    print(f"\n{'-'*20}END{'-'*20}\nBest accuracy ever of categorizer {best_acc*100:.2f}%")

if __name__ == "__main__":
    runCategorizer()
    # predictCategory()
    # findParams()
