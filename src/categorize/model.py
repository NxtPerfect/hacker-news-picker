from src.stats.stats import readStats, updateModel
import torch

from src.models.datasets import CategoryDataset
from src.models.models import CategorizerRNN, saveModel, loadModel, getTrainValidationTestDataloadersFromDataset
from src.database.db import DB_URL, loadData, saveNewData

EPOCHS = 25 # best accuracy using 25

def train(model, train_dataloader):
    model.to(model.device)

    loss = model.trainModel(EPOCHS, train_dataloader)

    print(f"Finished training .\n{'-' * 20}")

    return loss

def evaluate(model, validation_dataloader, loss):
    correct, total = model.evaluate(validation_dataloader, loss)

    print(f"Finished evaluating.\n{'-' * 20}")

    return model, correct, total

def testModel(model, test_dataloader):
    correct = 0
    total = 0

    with torch.no_grad():
        model.eval()
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct / total

    print(f"Test accuracy: {test_acc*100:.2f}% Correct/Total: {correct}/{total}")

    return test_acc, correct, total

def runModel():
    stats = readStats()
    best_acc, best_correct, best_wrong = getCategorizerStats(stats)

    print(f"\n{'#'*20} Running Categorizer {'#'*20}\n")

    model, acc, correct, total = runTraining()
    if acc > best_acc:
        saveBestModel(acc, best_acc, model)
        best_acc = round(acc, 4)
        best_correct = correct
        best_wrong = total - correct

    updateModel("categorizer", best_acc, best_correct, best_wrong)

    print(f"\n{'-'*20}END{'-'*20}\nBest accuracy of categorizer {best_acc*100:.2f}%")

def getCategorizerStats(stats):
    if stats == None:
        raise Exception("Couldn't read stats.")
    categorizer = stats["model"]["categorizer"]
    best_acc = categorizer["accuracy"]
    best_correct = categorizer["predict_correct"]
    best_wrong = categorizer["predict_wrong"]

    return best_acc, best_correct, best_wrong

def saveBestModel(acc, best_acc, model):
    print(f"Best accuracy so far {acc:.6f}. Saving...")

    isSuccessfullySavedModel = saveModel(model)
    if (isSuccessfullySavedModel):
        print("Successfully saved categorizer model.")

    return acc > best_acc

def runTraining():
    dataset = CategoryDataset(DB_URL)
    train_dataloader, val_dataloader, test_dataloader = getTrainValidationTestDataloadersFromDataset(dataset)

    vocab_size = len(dataset.tokenizer.vocab)
    output_dim = len(set(dataset.labels.numpy()))
    model = CategorizerRNN(vocab_size, output_dim)

    # Prevent gradient explosion by clipping max gradient
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    loss = train(model, train_dataloader)
    model, correct, total = evaluate(model, val_dataloader, loss)
    test_acc, correct, total = testModel(model, test_dataloader)

    return model, test_acc, correct, total


def predictCategory():
    print(f"[i] Started predicting category...")

    dataset = CategoryDataset(DB_URL, False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=24, shuffle=True)

    vocab_size = len(dataset.tokenizer.vocab)
    output_dim = len(set(dataset.labels.numpy()))-1
    model = CategorizerRNN(vocab_size, output_dim)
    model = loadModel(model)
    model.to(model.device)

    df = loadData(DB_URL)
    if type(df) == None:
        raise Exception("Loaded data from csv failed")

    new_df = model.predict(df, dataloader, dataset)
    # Save dataframe to test file
    saveNewData(new_df, model.categorized_database_path)

    print(f"[i] Finished predicting category.")

if __name__ == "__main__":
    runModel()
    # predictCategory()
    # findParams()
