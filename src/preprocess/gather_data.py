from bs4 import BeautifulSoup
import requests
import random
import pandas as pd
import time

from src.database.db import DB_URL

URL = "https://news.ycombinator.com/"
# then add number
PAGINATION = "https://news.ycombinator.com/?p="
# Randomize user_agent on startup or every 30 minutes
USER_AGENT_LIST = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.3", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.3", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.3"]
# Free proxies from https://free-proxy-list.net/#list
PROXIES_LIST = ["http://155.94.241.130:3128", "http://128.199.202.122:3128", "http://198.44.255.3:80", "http://31.220.78.244:80", "http://50.174.145.8:80", "http://58.234.116.197:80", "http://65.109.189.49:80"]
# Amount of pages to scrape
PAGES_AMOUNT = 20
# Time between requests, not sure if needed if i have proxies
# but better safe than sorry
TIMEOUT_TIME = 15

def parseArticle(article):
    link = article.find_next("a")
    age = article.find_next("span", class_="age")
    print(link["href"], link.contents[0], age)

    title = link.contents[0]
    link = link["href"]
    age = age["title"]
    return title, link, age

def saveArticles(df, title, link) -> pd.DataFrame | None:
    # if (df["Title"] == title).any() and (df["Link"] == link).any():
    # if title in df["Title"].isin([title]).any().any() and df["Link"].isin([link]).any().any():
    has_title = df["Title"].isin([title]).any().any()
    has_link = df["Link"].isin([link]).any().any()
    print(f"Has title: {has_title} has link: {has_link} for title: {title} and link: {link}")
    if has_title and has_link:
        return None

    df_new_rows = pd.DataFrame({
        'Title': [title],
        'Category': [None],
        'Link': [link],
        'Interest_Rating': [None],
        'Is_Fake_News': [None]
    })

    return df_new_rows

def run():
    for n in range(PAGES_AMOUNT):
        random_index = random.randint(0, len(USER_AGENT_LIST)-1)
        user_agent = USER_AGENT_LIST[random_index]
        random_index = random.randint(0, len(PROXIES_LIST)-1)
        proxy = {"http": PROXIES_LIST[random_index]}
        headers = {'User-Agent': user_agent}

        try:
            url = URL if n == 0 else PAGINATION + str(n)
            print(f"Currently looking at {url}")
            page = requests.get(url, headers=headers, proxies=proxy)
            soup = BeautifulSoup(page.content, 'html.parser')

            df = pd.read_csv(DB_URL)
            # new_dfs = []
            # td class="title" for title and <a href> inside of span class="titleline" for link
            # span class="age" for age of post
            articles = soup.find_all("span", class_="titleline")
            for article in articles:
                # Parse the article to get title, link and age
                title, link, age = parseArticle(article)
                # Append new article to existing dataframe
                new_df = saveArticles(df, title, link)
                # If new dataframe is empty, article exists in db, skip
                if type(new_df) != pd.DataFrame:
                    print("Already exists in the database, skipping...")
                    continue

                print("Adding article to database.")
                df = pd.concat([df, new_df])

            print("Saving df to file...")
            df.to_csv(DB_URL, index=False)
            print("Filed saved successfully.")

            # Timeout as to not spam requests and get blocked
            print(f"Sleeping for {TIMEOUT_TIME}seconds")
            time.sleep(TIMEOUT_TIME)
        except Exception as e:
            print("Failed to get page data.")
            print(e)
            return None

if __name__ == "__main__":
    run()
