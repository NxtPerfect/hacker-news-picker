import pandas as pd

DB_URL = "data/news.csv"

# Title / Category / Link / Interest_Rating / Is_Fake_News
def save(title: str, category: str, link: str, interest_rating: int, is_fake_news: bool):
    d = {'Title': title, 'Category': category, "Link": link, "Interest_Rating": interest_rating, "Is_Fake_News": is_fake_news}
    df = pd.DataFrame(data=d)
    df.to_csv(DB_URL, mode='a', header=False)
