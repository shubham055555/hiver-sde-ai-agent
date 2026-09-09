from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/twcs/twcs.csv")
OUTPUT_PATH = Path("data/processed/amazon_conversations.csv")

rows = []

for df in pd.read_csv(DATA_PATH, chunksize=100000):
    relevant = df[
        (df["author_id"] == "AmazonHelp")
        | (df["in_response_to_tweet_id"].notna())
    ]

    rows.append(
        relevant[
            [
                "tweet_id",
                "author_id",
                "inbound",
                "created_at",
                "text",
                "response_tweet_id",
                "in_response_to_tweet_id",
            ]
        ]
    )

tweets = pd.concat(rows, ignore_index=True)

amazon_ids = set(
    tweets.loc[
        tweets["author_id"] == "AmazonHelp",
        "tweet_id"
    ]
)

related_ids = set(amazon_ids)

for value in tweets["response_tweet_id"].dropna():
    for tweet_id in str(value).split(","):
        try:
            related_ids.add(int(float(tweet_id.strip())))
        except ValueError:
            pass

related_ids.update(
    tweets["in_response_to_tweet_id"]
    .dropna()
    .astype(int)
)

conversations = tweets[
    tweets["tweet_id"].isin(related_ids)
].copy()

conversations = conversations.sort_values(
    ["created_at", "tweet_id"]
)

conversations.to_csv(OUTPUT_PATH, index=False)

print("Tweets saved:", len(conversations))
print("Output:", OUTPUT_PATH)