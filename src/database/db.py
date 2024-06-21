import pandas as pd

DB_URL = "data/news.csv"

# Title / Category / Link / Interest_Rating
def saveData(df: pd.DataFrame, path="data/new_news.csv", mode='w'):
    try:
        print("[u] Saving data...")
        data = pd.DataFrame(data=df)
        data.to_csv(path, mode=mode)
        print("[u] Data saved successfully.")
        return True
    except Exception as e:
        print(f"Failed to save file.\n{e}")
        return False

def loadData(path="data/news.csv"):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"Failed to load data.\n{e}")
        return None
