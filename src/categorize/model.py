from scipy.integrate._ivp.rk import validate_tol
from src.stats.stats import readStats, updateModel
import torch

from src.models.datasets import CategoryDataset
from src.models.models import CategorizerRNN, getDevice, saveModel, loadModel, getTrainValidationTestDataloadersFromDataset
from src.database.db import DB_URL, loadData, saveData

EPOCHS = 25 # 25

def train(device, model, train_dataloader):
    model.to(device)

    loss = model.trainModel(EPOCHS, train_dataloader)

    print(f"Finished training .\n{'-' * 20}")

    return loss

def evaluate(model, validation_dataloader, loss):
    correct, total = model.evaluate(validation_dataloader, loss)

    print(f"Finished evaluating.\n{'-' * 20}")

    return model, correct, total

def testModel(model, test_dataloader, device):
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
    return test_acc, correct, total

# TODO: Split function to smaller functions
def runModel():
    best_acc = 0
    best_correct = 0
    best_wrong = 0
    try:
        print(f"\n{'#'*20} Running Categorizer {'#'*20}\n")
        stats = readStats()
        if stats == None:
            raise Exception("Couldn't read stats.")
        categorizer = stats["model"]["categorizer"]
        best_acc = categorizer["accuracy"]
        best_correct = categorizer["predict_correct"]
        best_wrong = categorizer["predict_wrong"]
    except Exception as e:
        print("[!] Failed to load stats.\n{e}")
        return None
    # for n in range(5):
        # print(f"\n{'#'*20} Running {n + 1} time out of 5 {'#'*20}\n")
    model, acc, correct, total = runTraining()
    if (acc > best_acc):
        print(f"Best accuracy so far {acc:.6f}. Saving...")
        if (saveModel(model)):
            print("Successfully saved categorizer model.")
        best_acc = acc
        best_correct = correct
        best_wrong = total - correct
    print(updateModel("categorizer", round(best_acc, 4), categorizer["feedback_correct"], categorizer["feedback_wrong"], best_correct, best_wrong))
    print(f"\n{'-'*20}END{'-'*20}\nBest accuracy of categorizer {best_acc*100:.2f}%")

# TODO: Split function to smaller functions
# TODO: Dataset depending on model
def runTraining():
    device = getDevice()
    # Load data
    dataset = CategoryDataset(DB_URL, 100)

    train_dataloader, val_dataloader, test_dataloader = getTrainValidationTestDataloadersFromDataset(dataset)

    # Prepare layers
    vocab_size = len(dataset.tokenizer.vocab)
    output_dim = len(set(dataset.labels.numpy()))

    # Create RNN
    model = CategorizerRNN(vocab_size, output_dim)

    # Train
    loss = train(device, model, train_dataloader)
    model, correct, total = evaluate(model, val_dataloader, loss)
    test_acc, correct, total = testModel(model, test_dataloader, device)

    return model, test_acc, correct, total


def predictCategory():
    device = getDevice()

    print(f"[i] Started predicting category...")

    dataset = CategoryDataset(DB_URL, 100)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)


    # Prepare layers
    vocab_size = len(dataset.tokenizer.vocab)
    output_dim = len(set(dataset.labels.numpy())) - 1 # Exclude null values

    # Create RNN
    model = CategorizerRNN(vocab_size, output_dim)
    model = loadModel(model)
    model.to(device)

    df = loadData()
    if df == None:
        raise Exception("Loaded data from csv failed")

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

if __name__ == "__main__":
    runModel()
    # predictCategory()
    # findParams()
