import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import json
import pandas as pd
import duckdb

# Load API key
load_dotenv()
API_KEY = os.getenv("API_KEY")  # Guardian API key

# Parameters
QUERY = "artificial intelligence"
BASE_URL = "https://content.guardianapis.com/search"
FROM_DATE = (datetime.today() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
TO_DATE = datetime.today().strftime("%Y-%m-%d")

PAGE_SIZE = 50  # max per request
TEST_MODE = True  # True = only fetch 10 articles for testing


all_articles = []
page = 1

while True:
    params = {
        "q": QUERY,
        "from-date": FROM_DATE,
        "to-date": TO_DATE,
        "page-size": PAGE_SIZE,
        "page": page,
        "order-by": "newest",
        "show-fields": "body,headline,webPublicationDate,shortUrl",
        "api-key": API_KEY,
    }

    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")

    data = response.json()
    results = data["response"]["results"]
    all_articles.extend(results)

    print(f"Fetched page {page} with {len(results)} articles")

    # Stop if TEST_MODE and we've got at least 10 articles
    if TEST_MODE and len(all_articles) >= 10:
        all_articles = all_articles[:10]  # trim to 10 for testing
        break

    # Stop if no more pages
    if page >= data["response"]["pages"]:
        break

    page += 1

# Save a JSON backup named by the pull date range
backup_path = f"../data/raw_articles_{FROM_DATE}_to_{TO_DATE}.json"
with open(backup_path, "w") as f:
    json.dump(all_articles, f, indent=2)
print(f"Saved backup to {backup_path}")
