from src.stats.stats import readStats, updateModel
import torch

from src.models.datasets import InterestDataset
from src.models.models import PredicterRNN, saveModel, loadModel
from src.database.db import DB_URL, loadData, saveData

EPOCHS = 25 # 50

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
    optimizer = torch.optim.Adam(model.parameters(), lr=model.LEARNING_RATE)
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
    output_dim = len(set(dataset.labels.numpy()))

    # TODO: Input size is equal to how many articles there are, which is bad bad
    # Create RNN
    model = PredicterRNN(vocab_size, output_dim)
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

    train_dataloader, val_dataloader, test_dataloader = getTrainValidationTestDataloadersFromDataset(dataset)

    # Prepare layers
    vocab_size = len(dataset.tokenizer.vocab)
    output_dim = len(set(dataset.labels.numpy()))

    # Create RNN
    model = PredicterRNN(vocab_size, output_dim)

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
    print(updateModel("predicter", round(best_acc, 4), predicter["feedback_correct"], predicter["feedback_wrong"], best_correct, best_wrong))
    print(f"\n{'-'*20}END{'-'*20}\nBest accuracy ever for predict model {best_acc*100:.2f}%")

if __name__ == "__main__":
    runPredicter()
    # predictInterest()
