from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/twcs/twcs.csv")
OUTPUT_PATH = Path("data/processed/amazon_threads.csv")

columns = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

tweets = pd.read_csv(DATA_PATH, usecols=columns)

tweets["parent_id"] = pd.to_numeric(
    tweets["in_response_to_tweet_id"],
    errors="coerce"
)

tweet_lookup = tweets.set_index("tweet_id").to_dict("index")

amazon_ids = set(
    tweets.loc[
        tweets["author_id"] == "AmazonHelp",
        "tweet_id"
    ]
)

conversation_cache = {}

def find_root(tweet_id):
    if tweet_id in conversation_cache:
        return conversation_cache[tweet_id]

    current = tweet_id
    visited = set()

    while current in tweet_lookup:
        if current in visited:
            break

        visited.add(current)

        parent = tweet_lookup[current]["parent_id"]

        if pd.isna(parent):
            break

        parent = int(parent)

        if parent not in tweet_lookup:
            break

        current = parent

    root = current

    for item in visited:
        conversation_cache[item] = root

    return root

amazon_threads = []

for tweet_id in amazon_ids:
    root = find_root(tweet_id)

    if root in tweet_lookup:
        row = tweet_lookup[root]

        amazon_threads.append(
            {
                "conversation_id": root,
                "root_author_id": row["author_id"],
                "root_text": row["text"],
                "root_created_at": row["created_at"],
            }
        )

threads = pd.DataFrame(amazon_threads).drop_duplicates(
    "conversation_id"
)

threads.to_csv(OUTPUT_PATH, index=False)

print("AmazonHelp conversations:", len(threads))
print("Output:", OUTPUT_PATH)