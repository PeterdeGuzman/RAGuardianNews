import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import json
from dateutil.relativedelta import relativedelta


# Load API key
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Parameters
QUERIES = ["artificial intelligence", "AI", "GenAI", "generative AI"]

BASE_URL = "https://content.guardianapis.com/search"
FROM_DATE = (datetime.today() - relativedelta(years=10)).strftime("%Y-%m-%d")
TO_DATE = datetime.today().strftime("%Y-%m-%d")

PAGE_SIZE = 50
TEST_MODE = False  # change to True if you want to limit results


all_articles = []

for QUERY in QUERIES:
    print(f"\nStarting query: {QUERY}")
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

        # Tag each article with the search term
        for article in results:
            article["search_term"] = QUERY

        all_articles.extend(results)

        print(f"Fetched page {page} with {len(results)} articles")

        # Optional test limit
        # if TEST_MODE and len(all_articles) >= 20:
        #     break

        if page >= data["response"]["pages"]:
            break

        page += 1

# ✅ Deduplicate by article ID
unique_articles = {}
for article in all_articles:
    unique_articles[article["id"]] = article

final_articles = list(unique_articles.values())

print(f"\nTotal articles after deduplication: {len(final_articles)}")

# Save JSON backup
backup_path = f"../data/raw_articles_{FROM_DATE}_to_{TO_DATE}.json"
with open(backup_path, "w") as f:
    json.dump(final_articles, f, indent=2)

print(f"Saved backup to {backup_path}")
