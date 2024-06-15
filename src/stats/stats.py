from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

YAML_PATH = "stats.yaml"

def updateModel(accuracy: float, feedback_correct: int, feedback_wrong: int, predict_correct: int, predict_wrong: int):
    # accuracy: 0 # in percentage * 100
    # feedback_correct: 0 # feedbacks about correct prediction
    # feedback_wrong: 0 # feedbacks about wrong prediction 
    # predict_correct: 0 # predicted correct during training
    # predict_wrong: 0 # predicted wrong during training
    with open(YAML_PATH, "r") as f:
        data = load(f, Loader=Loader)
        new_data = {'accuracy': accuracy, 'feedback_correct': feedback_correct, 'feedback_wrong': feedback_wrong, 'predict_correct': predict_correct, 'predict_wrong': predict_wrong}
        try:
            data["model"] = new_data
            with open(YAML_PATH, "w") as fw:
                dump(data, fw, Dumper=Dumper)
                print("Successfully updated model data.")
                return True
        except Exception as e:
            print("Failed to update model data.")
            print(e)
            return False

def updateDatabase(articles: int, articles_fetched: int, categories_count: int, categories_list: list):
    # articles: 0 # unique articles in database
    # articles_fetched: 0 # articles fetched from website
    # categories_count: 0 # unique categories in database
    # categories_list: [] # unique categories in database
    with open(YAML_PATH, "r") as f:
        data = load(f, Loader=Loader)
        new_data = {'articles': articles, 'articles_fetched': articles_fetched, 'categories_count': categories_count, 'categories_list': categories_list}
        try:
            data["database"] = new_data
            with open(YAML_PATH, "w") as fw:
                dump(data, fw, Dumper=Dumper)
                print("Successfully updated database data.")
                return True
        except Exception as e:
            print("Failed to update database data.")
            print(e)
            return False

def readStats():
    try:
        with open(YAML_PATH, "r") as f:
            data = load(f, Loader=Loader)
            return data
    except Exception as e:
        print("Failed to read stats.")
        print(e)
        return None
