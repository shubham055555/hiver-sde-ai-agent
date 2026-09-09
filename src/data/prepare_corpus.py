from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/amazon_thread_messages.csv")
OUTPUT_PATH = Path("data/processed/amazon_corpus.csv")

df = pd.read_csv(INPUT_PATH)

stats = df.groupby("conversation_id").agg(
    messages=("tweet_id", "count"),
    customer_messages=("inbound", "sum"),
    amazon_messages=("inbound", lambda x: (~x).sum()),
)

valid_ids = stats[
    (stats["customer_messages"] >= 1)
    & (stats["amazon_messages"] >= 1)
    & (stats["messages"] <= 20)
].index

corpus = df[df["conversation_id"].isin(valid_ids)].copy()

corpus = corpus.sort_values(
    ["conversation_id", "created_at", "tweet_id"]
)

corpus.to_csv(OUTPUT_PATH, index=False)

print("Original conversations:", len(stats))
print("Usable conversations:", len(valid_ids))
print("Removed conversations:", len(stats) - len(valid_ids))
print("Messages in corpus:", len(corpus))
print("Output:", OUTPUT_PATH)