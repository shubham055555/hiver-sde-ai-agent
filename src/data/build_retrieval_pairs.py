import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/processed/amazon_corpus_clean.csv")
OUTPUT_PATH = Path("data/processed/retrieval_pairs.csv")


df = pd.read_csv(DATA_PATH)

df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df = df.sort_values(["conversation_id", "created_at"])

pairs = []

for conversation_id, group in df.groupby("conversation_id"):
    messages = group.to_dict("records")

    for i in range(len(messages) - 1):
        customer = messages[i]
        reply = messages[i + 1]

        if customer["inbound"] is True and reply["inbound"] is False:
            if pd.notna(customer["clean_text"]) and pd.notna(reply["clean_text"]):
                pairs.append({
                    "conversation_id": conversation_id,
                    "customer_tweet_id": customer["tweet_id"],
                    "reply_tweet_id": reply["tweet_id"],
                    "customer_text": customer["text"],
                    "customer_clean_text": customer["clean_text"],
                    "amazon_reply": reply["text"],
                    "amazon_reply_clean": reply["clean_text"]
                })


pairs_df = pd.DataFrame(pairs)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
pairs_df.to_csv(OUTPUT_PATH, index=False)

print(f"Pairs created: {len(pairs_df)}")
print(f"Saved to: {OUTPUT_PATH}")