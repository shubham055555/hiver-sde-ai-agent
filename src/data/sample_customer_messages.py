from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/amazon_corpus.csv")

df = pd.read_csv(
    INPUT_PATH,
    usecols=["conversation_id", "tweet_id", "inbound", "text"]
)

customer = df[df["inbound"] == True].copy()

sample = customer.sample(
    n=min(500, len(customer)),
    random_state=42
)

sample.to_csv(
    "data/processed/customer_sample.csv",
    index=False
)

print("Customer messages:", len(customer))
print("Sample saved:", len(sample))
print("Output: data/processed/customer_sample.csv")