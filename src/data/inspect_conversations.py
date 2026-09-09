from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/twcs/twcs.csv")

tweets = pd.read_csv(DATA_PATH)

amazon = tweets[
    (tweets["author_id"] == "AmazonHelp") |
    (tweets["in_response_to_tweet_id"].isin(
        tweets.loc[tweets["author_id"] == "AmazonHelp", "tweet_id"]
    ))
].copy()

amazon = amazon.sort_values("created_at")

print("AmazonHelp related tweets:", len(amazon))
print()

for _, row in amazon.head(40).iterrows():
    direction = "CUSTOMER" if row["inbound"] else "AMAZON"
    print(f"{direction} | {row['tweet_id']}")
    print(row["text"])
    print()