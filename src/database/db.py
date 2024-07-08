import pandas as pd

DB_URL = "data/news.csv"

# Title / Category / Link / Interest_Rating
def saveData(df: pd.DataFrame, path="data/news.csv", mode='w'):
    try:
        print("[u] Saving data...")
        data = pd.DataFrame(data=df)
        data.to_csv(path, index=False, mode=mode)
        print("[u] Data saved successfully.")
        return True
    except Exception as e:
        print(f"!E! Failed to save file.\n{e}")
        return False

def loadData(path="data/news.csv"):
    try:
        df = pd.read_csv(path, skip_blank_lines=True).dropna(how="all")
        return df
    except Exception as e:
        print(f"!E! Failed to load data.\n{e}")
        return None
