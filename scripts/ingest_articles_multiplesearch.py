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
TEST_MODE = False

# Collect all articles across queries, tracking which terms matched each ID
articles_by_id = {}  # id -> article dict
terms_by_id = {}  # id -> set of matching search terms

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

        for article in results:
            article_id = article["id"]
            if article_id not in articles_by_id:
                articles_by_id[article_id] = article
                terms_by_id[article_id] = set()
            terms_by_id[article_id].add(QUERY)

        print(f"Fetched page {page} with {len(results)} articles")

        if TEST_MODE and len(articles_by_id) >= 20:
            break

        if page >= data["response"]["pages"]:
            break

        page += 1

# Attach the comma-separated search_terms field to each article
final_articles = []
for article_id, article in articles_by_id.items():
    article["search_terms"] = ", ".join(sorted(terms_by_id[article_id]))
    final_articles.append(article)

print(f"\nTotal unique articles collected: {len(final_articles)}")
multi_match = sum(1 for a in final_articles if "," in a["search_terms"])
print(f"Articles matched by more than one search term: {multi_match}")

backup_path = f"../data/raw_articles_{FROM_DATE}_to_{TO_DATE}.json"
with open(backup_path, "w") as f:
    json.dump(final_articles, f, indent=2)

print(f"Saved backup to {backup_path}")
