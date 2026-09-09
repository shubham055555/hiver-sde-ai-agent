from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/twcs/twcs.csv")

columns = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

frames = []

for df in pd.read_csv(DATA_PATH, usecols=columns, chunksize=100000):
    relevant = df[
        (df["author_id"] == "AmazonHelp")
        | (df["in_response_to_tweet_id"].notna())
    ]
    frames.append(relevant)

tweets = pd.concat(frames, ignore_index=True)

tweets["parent_id"] = pd.to_numeric(
    tweets["in_response_to_tweet_id"],
    errors="coerce"
)

tweet_ids = set(tweets["tweet_id"])

amazon = tweets[tweets["author_id"] == "AmazonHelp"]

amazon_with_parent = amazon[
    amazon["parent_id"].isin(tweet_ids)
]

print("AmazonHelp tweets:", len(amazon))
print("AmazonHelp tweets with known parent:", len(amazon_with_parent))

print()
print("Sample parent relationships:")

sample = amazon_with_parent[
    ["tweet_id", "parent_id", "text"]
].head(20)

print(sample.to_string(index=False))