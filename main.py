# from src.categorize.model import predictCategory, runCategorizer
# from src.predict.model import predictInterest, runPredicter
from src.categorize.model import predictCategory, runModel
from src.predict.model import predictInterest, runPredicter
from src.preprocess.gather_data import asyncRunScraper

def runMain():
    # Preprocess data
    asyncRunScraper()
    
    # Run categorizer
    runModel()

    # Fill out data
    predictCategory()

    # Run predicter
    runPredicter()

    # Fill out interest data
    predictInterest()

if __name__ == "__main__":
    runMain()
