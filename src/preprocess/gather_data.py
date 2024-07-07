#!/bin/env python
from bs4 import BeautifulSoup
import requests
import random
import pandas as pd
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import cpu_count

from src.database.db import DB_URL, saveData
from src.stats.stats import readStats, updateDatabase

# Website url to scrape with pagination
URL = "https://news.ycombinator.com/?p="

# Randomize user_agent on startup or every 30 minutes
USER_AGENT_LIST = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.3",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.3",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.3"
        ]

# Free proxies from https://free-proxy-list.net/#list
# 9 / 20
PROXIES_LIST = [
        "http://155.94.241.130:3128",
        "http://128.199.202.122:3128",
        "http://198.44.255.3:80",
        "http://31.220.78.244:80",
        "http://50.174.145.8:80",
        "http://58.234.116.197:80",
        "http://65.109.189.49:80",
        "http://123.30.154.171:7777",
        "http://50.168.163.177:80",
        "http://62.99.138.162:80",
        "http://91.92.155.207:3128",
        "http://85.8.68.2:80",
        "http://47.74.152.29:8888",
        "http://83.1.176.118:80",
        "http://167.102.133.111:80",
        "http://50.207.199.84:80",
        "http://103.163.51.254:80",
        "http://50.172.75.126:80",
        "http://211.128.96.206:80",
        "http://51.254.78.223:80",
        ]

# Amount of pages to scrape
PAGES_AMOUNT = 20

def parseArticle(article, debug=False):
    # Find link
    link = article.find_next("a")

    # Find title
    title = link.contents[0]

    # Get link url
    link = link["href"]
    print(f"[i] Found title: '{title}' with link: '{link}'") if debug else None
    return title, link

def saveArticles(df, title, link) -> pd.DataFrame | None:
    # Check if title and link in db
    # if both in db, don't save
    has_title = df["Title"].isin([title]).any().any()
    has_link = df["Link"].isin([link]).any().any()
    if has_title and has_link:
        return None

    return pd.DataFrame({
        'Title': [title],
        'Category': [None],
        'Link': [link],
        'Interest_Rating': [None],
    })

def runScraperAsync():
    urls = list(f"{URL}{n+1}" for n in range(PAGES_AMOUNT))
    thread_count = min(len(urls), cpu_count())

    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = [executor.submit(requestArticle, url) for url in urls]

    new_articles_count = []
    df = pd.read_csv(DB_URL)

    # Get each article as it's done
    for future in as_completed(futures):
        res = future.result()

        if res[0]["Title"].isnull().any() or res[0]["Link"].isnull().any():
            continue
        df = pd.concat([df, res[0]])

        new_articles_count.append(res[1])

    new_article_count = 0
    new_article_count += sum(new_articles_count)

    # Save article
    print("\n")
    saveData(df)

    # Update stats
    stats = readStats()
    articles_count = new_article_count + stats["database"]["articles"]

    updateDatabase(articles_count, stats["database"]["categories_count"], stats["database"]["categories_list"])

    print(f"Found {new_article_count} new articles")

def requestArticle(url, debug=False) -> tuple[pd.DataFrame, int] | tuple[None, None]:
    try:
        new_article_count = 0

        random_index = random.randint(0, len(USER_AGENT_LIST)-1)
        user_agent = USER_AGENT_LIST[random_index]
        random_index = random.randint(0, len(PROXIES_LIST)-1)
        proxy = {"http": PROXIES_LIST[random_index]}
        headers = {'User-Agent': user_agent}

        page = requests.get(url, headers=headers, proxies=proxy)
        soup = BeautifulSoup(page.content, 'html.parser')

        df = pd.DataFrame({
            'Title': [None],
            'Category': [None],
            'Link': [None],
            'Interest_Rating': [None],
        })

        # Find article
        articles = soup.find_all("span", class_="titleline")
        for i, article in enumerate(articles):
            percent = ((i+1) / len(articles)) * 100.0
            print(f"\rParsing article {i+1} out of {len(articles)} |{'█'*(i+1)}{'-'*(len(articles) - (i+1))}| {percent:.2f}%", end='')
            # Parse the article to get title, link and age
            print("[i] Parsing article...") if debug else None
            title, link = parseArticle(article)
            print("[i] Article successfully parsed.") if debug else None

            # Append new article to existing dataframe
            print("[u] Saving article as dataframe...") if debug else None
            new_df = saveArticles(df, title, link)
            print("[u] Article successfully saved as dataframe.") if debug else None

            # Create new dataframe and concat to existing
            if type(new_df) != pd.DataFrame:
                print(f"Article in database") if debug else None
                continue

            df = pd.concat([df, new_df])
            new_article_count += 1

        return df, new_article_count

    except Exception as e:
        print(f"!e! Request failed due to error {e}")
        return None, None

def runScraper():
    new_article_count = 0
    for n in range(PAGES_AMOUNT):
        # Prepare headers and proxies
        random_index = random.randint(0, len(USER_AGENT_LIST)-1)
        user_agent = USER_AGENT_LIST[random_index]
        random_index = random.randint(0, len(PROXIES_LIST)-1)
        proxy = {"http": PROXIES_LIST[random_index]}
        headers = {'User-Agent': user_agent}

        stats = readStats()

        try:
            url = f"{URL}{n+1}"
            print(f"\n{30*'#'}\n")
            print(f"[i] Currently looking at {url}")
            page = requests.get(url, headers=headers, proxies=proxy)
            soup = BeautifulSoup(page.content, 'html.parser')

            # Read db to update articles count
            df = pd.read_csv(DB_URL)
            articles_count = stats["database"]["articles"]

            # Find article
            articles = soup.find_all("span", class_="titleline")
            for article in articles:
                # Parse the article to get title, link and age
                print("[i] Parsing article...")
                title, link = parseArticle(article)
                print("[i] Article successfully parsed.")

                # Append new article to existing dataframe
                print("[u] Saving article as dataframe...")
                new_df = saveArticles(df, title, link)
                print("[u] Article successfully saved as dataframe.")

                # If new dataframe is empty, article exists in db, skip
                if type(new_df) != pd.DataFrame:
                    print("[s] Already exists in the database, skipping...")
                    continue

                print("[u] Adding article to database.")
                df = pd.concat([df, new_df])
                articles_count += 1
                new_article_count += 1

            print("[u] Saving df to file...")
            saveData(df, DB_URL)
            print("[u] Filed saved successfully.")

            print("[u] Updating stats...")
            updateDatabase(articles_count, stats["database"]["categories_count"], stats["database"]["categories_list"])
            print("[u] Stats updated.")
        except Exception as e:
            print("Failed to get page data.")
            print(e)
            return None
    print(f"\n{'-'*40}\nFinished running process with {new_article_count} new articles")

if __name__ == "__main__":
    # start = perf_counter()
    # runScraper()
    # stop = perf_counter()
    # print(f"\n\nTime took {(stop-start):.2f}s for {PAGES_AMOUNT} pages.")

    start = perf_counter()
    runScraperAsync()
    stop = perf_counter()
    print(f"\n\nTime took async {(stop-start):.2f}s for {PAGES_AMOUNT} pages.")
