from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/twcs/twcs.csv")
OUTPUT_PATH = Path("data/processed/amazon_thread_messages.csv")

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

amazon_conversation_ids = set()

for tweet_id in amazon_ids:
    amazon_conversation_ids.add(find_root(tweet_id))

tweets["conversation_id"] = tweets["tweet_id"].map(
    lambda tweet_id: conversation_cache.get(
        tweet_id,
        find_root(tweet_id)
    )
)

amazon_threads = tweets[
    tweets["conversation_id"].isin(amazon_conversation_ids)
].copy()

amazon_threads = amazon_threads.sort_values(
    ["conversation_id", "created_at", "tweet_id"]
)

amazon_threads[
    [
        "conversation_id",
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]
].to_csv(OUTPUT_PATH, index=False)

print("Conversation messages:", len(amazon_threads))
print("Conversations:", amazon_threads["conversation_id"].nunique())
print("Output:", OUTPUT_PATH)