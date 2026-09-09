from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/amazon_corpus.csv")
SAMPLE_PATH = Path("data/processed/customer_sample.csv")

df = pd.read_csv(INPUT_PATH)

sample = pd.read_csv(SAMPLE_PATH)

df["created_at"] = pd.to_datetime(df["created_at"])
sample_ids = set(sample["tweet_id"])

rows = []

for _, row in sample.iterrows():
    conversation = df[
        df["conversation_id"] == row["conversation_id"]
    ].sort_values(["created_at", "tweet_id"])

    position = conversation.index[
        conversation["tweet_id"] == row["tweet_id"]
    ]

    if len(position) == 0:
        continue

    position = conversation.index.get_loc(position[0])
    following = conversation.iloc[position + 1:]

    amazon_reply = following[
        following["inbound"] == False
    ]

    reply = ""
    if len(amazon_reply) > 0:
        reply = amazon_reply.iloc[0]["text"]

    rows.append(
        {
            "conversation_id": row["conversation_id"],
            "customer_message": row["text"],
            "amazon_reply": reply,
        }
    )

result = pd.DataFrame(rows)

result.to_csv(
    "data/processed/intent_review_sample.csv",
    index=False
)

print("Examples:", len(result))
print("Output: data/processed/intent_review_sample.csv")

for _, row in result.head(30).iterrows():
    print()
    print("CUSTOMER:", row["customer_message"])
    print("AMAZON:", row["amazon_reply"])