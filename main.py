from src.predict.model import predictInterest, runPredicter
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

    # Fill out interest data
    predictInterest()

if __name__ == "__main__":
    runMain()
