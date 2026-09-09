from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/twcs/twcs.csv")

target_brands = {
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "Tesco",
    "AmericanAir",
    "TMobileHelp",
    "comcastcares",
}

brand_tweet_ids = {brand: set() for brand in target_brands}

for df in pd.read_csv(DATA_PATH, chunksize=100000):
    outbound = df[df["inbound"] == False]

    for brand in target_brands:
        ids = outbound.loc[
            outbound["author_id"] == brand,
            "tweet_id"
        ]

        brand_tweet_ids[brand].update(ids.tolist())

print("Brand tweets collected")

stats = {}

for brand in target_brands:
    brand_ids = brand_tweet_ids[brand]
    inbound_count = 0

    for df in pd.read_csv(DATA_PATH, chunksize=100000):
        inbound = df[df["inbound"] == True]
        replies = inbound["in_response_to_tweet_id"].isin(brand_ids)
        inbound_count += int(replies.sum())

    stats[brand] = {
        "brand_tweets": len(brand_ids),
        "customer_replies": inbound_count,
    }

result = pd.DataFrame(stats).T
result = result.sort_values("customer_replies", ascending=False)

print(result.to_string())