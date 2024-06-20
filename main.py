from src.predict.model import runPredicter
from src.preprocess.gather_data import runScraper
from src.categorize.model import runCategorizer


def runMain():
    # Preprocess data
    runScraper()
    
    # Run categorizer
    runCategorizer()

    # Run predicter
    runPredicter()

    # The runs should instead
    # fill out the dataframe

if __name__ == "__main__":
    runMain()
