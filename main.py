from src.predict.model import runPredicter
from src.preprocess.gather_data import runScraper
from src.categorize.model import predictCategory, runCategorizer

def runMain():
    # Preprocess data
    runScraper()
    
    # Run categorizer
    runCategorizer()

    # Fill out data
    predictCategory()

    # Run predicter
    runPredicter()

    # The runs should instead
    # fill out the dataframe

if __name__ == "__main__":
    runMain()
