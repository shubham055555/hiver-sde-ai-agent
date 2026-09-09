from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/amazon_thread_messages.csv")

df = pd.read_csv(DATA_PATH)

conversation_stats = df.groupby("conversation_id").agg(
    messages=("tweet_id", "count"),
    customer_messages=("inbound", "sum"),
    amazon_messages=("inbound", lambda x: (~x).sum()),
    unique_authors=("author_id", "nunique"),
)

conversation_stats["has_customer"] = (
    conversation_stats["customer_messages"] > 0
)

conversation_stats["has_amazon"] = (
    conversation_stats["amazon_messages"] > 0
)

conversation_stats["multi_turn"] = (
    conversation_stats["messages"] >= 3
)

print("Total conversations:", len(conversation_stats))
print(
    "With customer messages:",
    conversation_stats["has_customer"].sum()
)
print(
    "With AmazonHelp messages:",
    conversation_stats["has_amazon"].sum()
)
print(
    "Multi-turn conversations:",
    conversation_stats["multi_turn"].sum()
)

print()
print("Message count distribution:")
print(conversation_stats["messages"].describe())

print()
print("Customer message distribution:")
print(conversation_stats["customer_messages"].describe())