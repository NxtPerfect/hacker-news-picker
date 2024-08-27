#!/bin/env python
from bs4 import BeautifulSoup
import lxml # better html parser
import cchardet # speeds up beautifulsoup
import requests
import random
import pandas as pd
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import cpu_count, path

from src.database.db import DB_URL, appendDataToExistingFile
from src.stats.stats import readStats, updateDatabase

# Website url to scrape with pagination
URL = "https://news.ycombinator.com/"
PAGINATION = "?p="

# Randomize user_agent on startup or every 30 minutes
USER_AGENT_LIST = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.3",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.3",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.3"
        ]

# Free proxies from https://free-proxy-list.net/#list
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
PAGES_AMOUNT = 25

def parseArticle(article, debug=False):
    if debug: print("[i] Parsing article...")
    # Find link
    link = article.find_next("a")

    # Find title
    title = link.contents[0]

    # Get link url
    link = link["href"]
    if link.startswith('item?id='): link = f'{URL}{link}'
    if debug: print(f"[i] Found title: '{title}' with link: '{link}'")
    if debug: print("[i] Article successfully parsed.")
    return title, link

def articleIntoDataframe(df, title, link, debug=False) -> pd.DataFrame | None:
    if debug: print("[u] Saving article as dataframe...")
    # Read csv, check if link already in database
    if link in df["Link"].values:
        return None

    if debug: print("[u] Article successfully saved as dataframe.")

    return pd.DataFrame({
        'Title': [title],
        'Category': [None],
        'Link': [link],
        'Interest_Rating': [None],
    })

def getUrls(pagesAmount):
    return list(f"{URL}{PAGINATION}{n+1}" for n in range(pagesAmount))

def filterDataframe(all_dfs):
    return [df for df in all_dfs if not df.empty and not df.isna().all().all()]

def updateStats(all_dfs):
    # new_article_count = sum(new_articles_count)
    new_article_count = len(all_dfs)
    stats = readStats()
    articles_count = new_article_count + stats["database"]["articles"]

    updateDatabase(articles_count, stats["database"]["categories_count"], stats["database"]["categories_list"])
    print(f"Found {new_article_count} new articles." if new_article_count > 0 else "No new articles found.")

def saveResultsToCsv(results):
    all_dfs = []

    for res in results:
        if res[0]["Title"].isnull().any() or res[0]["Link"].isnull().any():
            continue
        all_dfs.append(res[0])

    filtered_dfs = filterDataframe(all_dfs)
    appendDataToExistingFile(filtered_dfs, DB_URL)

    updateStats(all_dfs)

def getCompletedThreads(futures):
    results = []
    # Get each article as it's done
    for future in as_completed(futures):
        res = future.result()
        if res[0] is not None:
            results.append(res)
    return results

def runThreads(urls, thread_count):
    df = pd.read_csv(DB_URL)
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = [executor.submit(requestArticle, session, url, df) for url in urls]

    results = getCompletedThreads(futures)
    return results

def asyncRunScraper():
    urls = getUrls(PAGES_AMOUNT)
    thread_count = min(len(urls), cpu_count())

    results = runThreads(urls, thread_count)
    saveResultsToCsv(results)

def getRandomProxy():
    random_index = random.randint(0, len(PROXIES_LIST)-1)
    randomProxy = PROXIES_LIST[random_index]
    return randomProxy

def getRandomUserAgent():
    random_index = random.randint(0, len(USER_AGENT_LIST)-1)
    randomUserAgent = USER_AGENT_LIST[random_index]
    return randomUserAgent

def prepareSession(session):
    proxy = getRandomProxy()
    session.proxies.update({"http": proxy})

    userAgent = getRandomUserAgent()
    session.headers.update({
        'User-Agent': userAgent
    })
    return session

def requestPage(url, session):
    page = session.get(url)
    return page

def parsePageForArticles(page):
    soup = BeautifulSoup(page.content, 'lxml')
    articles = soup.find_all("span", class_="titleline")
    return articles

def parseArticlesAndSaveToDataframe(articles, debug=False):
    new_df = None
    all_dfs = []
    df = pd.read_csv(DB_URL)

    for i, article in enumerate(articles):
        percent = ((i+1) / len(articles)) * 100.0
        print(f"\rParsing article {i+1} out of {len(articles)} |{'█'*(i+1)}{'-'*(len(articles) - (i+1))}| {percent:.2f}%", end='')
        # Parse the article to get title, link and age
        title, link = parseArticle(article)

        # Append new article to existing dataframe
        new_df = articleIntoDataframe(df, title, link)

        # Create new dataframe and concat to existing
        if not isinstance(new_df, pd.DataFrame):
            if debug: print(f"[i] Article already in database.")
            continue

        all_dfs.append(new_df)

    new_article_count = len(all_dfs)

    if len(all_dfs): new_df = pd.concat(all_dfs, ignore_index=True)
    return new_df, new_article_count


def requestArticle(session, url, df, debug=False) -> tuple[pd.DataFrame, int] | tuple[None, None]:
    session = prepareSession(session)
    try:
        page = requestPage(url, session)
        articles = parsePageForArticles(page)

        new_df, new_article_count = parseArticlesAndSaveToDataframe(articles)

        return new_df, new_article_count
    except Exception as e:
        print(f"!e! Request failed due to error {e}")
        return None, None

def time(func):
    start = perf_counter()
    func()
    stop = perf_counter()
    return stop-start

if __name__ == "__main__":
    # print(f"File size before scraping {path.getsize(DB_URL):,} bytes")
    runningTime = time(asyncRunScraper)
    # print(f"\nFile size after scraping {path.getsize(DB_URL):,} bytes")
    print(f"\n\nTime took async {(runningTime):.2f}s for {PAGES_AMOUNT} pages.")
