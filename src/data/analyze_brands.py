from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/twcs/twcs.csv")

brand_counts = {}

for df in pd.read_csv(DATA_PATH, chunksize=100000):
    outbound = df[df["inbound"] == False]
    counts = outbound["author_id"].value_counts()

    for author, count in counts.items():
        brand_counts[author] = brand_counts.get(author, 0) + count

brands = (
    pd.Series(brand_counts, name="outbound_tweets")
    .sort_values(ascending=False)
    .head(30)
)

print(brands.to_string())