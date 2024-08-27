import pandas as pd

DB_URL = "data/news.csv"

def appendDataToExistingFile(df: pd.DataFrame, path: str):
    currentDf = loadData(path)
    currentDf = pd.concat([currentDf] + df, ignore_index=True)
    try:
        print("[u] Saving data...")
        data = pd.DataFrame(data=currentDf)
        data.to_csv(path, index=False, mode='w')
        print("[u] Data saved successfully.")
        return True
    except Exception as e:
        print(f"!E! Failed to save file with error: {e}")
        return False

def compressedAppendExistingDataToFile(df: pd.DataFrame, path: str):
    currentDf = loadData(path)
    currentDf = pd.concat([currentDf] + df, ignore_index=True)
    try:
        print("[u] Saving data...")
        data = pd.DataFrame(data=df)
        data.to_csv(path, index=False, mode='w', compression='gzip')
        print("[u] Data saved successfully.")
        return True
    except Exception as e:
        print(f"!E! Failed to save compressed file with error: {e}")
        return False

def saveNewData(df: pd.DataFrame, path:str):
    try:
        print("[u] Saving data...")
        data = pd.DataFrame(data=df)
        data.to_csv(path, index=False, mode='w', compression='gzip')
        print("[u] Data saved successfully.")
        return True
    except Exception as e:
        print(f"!E! Failed to save compressed file with error: {e}")
        return False

def loadDataWithoutNulls(path: str):
    try:
        df = pd.read_csv(path, skip_blank_lines=True).dropna(how="all")
        return df
    except Exception as e:
        print(f"!E! Failed to load data.\n{e}")
        return None

def loadData(path: str):
    try:
        df = pd.read_csv(path, skip_blank_lines=True)
        return df
    except Exception as e:
        print(f"!E! Failed to load data.\n{e}")
        return None
