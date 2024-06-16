import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from src.categorize.dataset import CategoryDataset
from src.database.db import DB_URL

CATEGORIZE_MODEL_PATH = "model/categorize"
EPOCHS = 50

def createModel():
    pass
    model = torch.nn.RNN(10, 20, 15, dropout=0.2) # Random parameters as placeholders
    return model

def train(model):
    pass
    data = CategoryDataset()

    # My labels are 1d, with features being 2d, both are tensors
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    print(features_train, labels_train)

    # If cuda is available use it
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    num_epochs = 0
    loss = 0

    # Training the model
    while num_epochs < EPOCHS:
        for inputs, labels in zip(features_train, labels_train):
            inputs, labels = inputs.to(device), labels.to(device)

            print(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        num_epochs += 1
        print(f"Epoch: {num_epochs}, loss: {loss.item():.6f}")

    # Evaluation for training data
    model.eval()
    with torch.no_grad():
        validation_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in zip(features_test, labels_test):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            validation_loss += loss.item()


    scheduler.step()

    accuracy = correct / total
    avg_validation_loss = validation_loss / len(test_dataloader_emnist)

    print(f"\n{'-' * 20}\nEMNIST:\n\nTest accuracy: {accuracy * 100:.2f}%, with average validation loss: {avg_validation_loss:.6f}.")

    print(f"Finished testing EMNIST.\n{'-' * 20}\nTesting on SEMEION dataset...")
    with torch.no_grad():
        validation_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in test_dataloader_semeion:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            validation_loss += loss.item()

    scheduler.step()

    accuracy = correct / total
    avg_validation_loss = validation_loss / len(test_dataloader_semeion)

    return features_train, features_test, labels_train, labels_test

def getData():
    pass

# Save model
def saveModel(model) -> bool:
    pass
    try:
        torch.save(model.state_dict(), CATEGORIZE_MODEL_PATH)
        return True
    except Exception as e:
        print("Failed to save model.")
        print(e)
        return False

def loadModel(model) -> torch.nn.RNN:
    pass
    model.load_state_dict(torch.load(CATEGORIZE_MODEL_PATH))
    model.eval()
    return model

def run():
    model = createModel()
    _ = train(model)
    print("Trained")
    return

if __name__ == "__main__":
    run()
