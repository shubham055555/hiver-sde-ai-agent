from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/amazon_corpus_clean.csv")

df = pd.read_csv(
    INPUT_PATH,
    usecols=["inbound", "clean_text"]
)

customer = df[df["inbound"] == True].copy()
text = customer["clean_text"].fillna("").str.lower()

rules = {
    "delivery_issue": [
        "delivery",
        "delivered",
        "shipping",
        "package",
        "tracking",
        "arrive",
        "carrier",
    ],
    "order_management": [
        "order",
        "cancel order",
        "order number",
        "order details",
    ],
    "returns_refunds": [
        "refund",
        "return",
        "replacement",
        "money back",
    ],
    "payment_billing": [
        "payment",
        "pay",
        "credit card",
        "debit card",
        "charge",
        "balance",
    ],
    "prime_subscription": [
        "prime membership",
        "prime member",
        "amazon prime",
        "prime subscription",
    ],
    "account_access": [
        "account",
        "sign in",
        "login",
        "password",
        "locked account",
    ],
    "product_technical": [
        "doesn't work",
        "not working",
        "error",
        "app",
        "kindle",
        "echo",
        "fire tv",
    ],
    "product_information": [
        "available",
        "availability",
        "feature",
        "compatible",
        "language",
    ],
    "seller_review": [
        "review",
        "seller feedback",
        "feedback",
    ],
    "support_escalation": [
        "customer service",
        "customer support",
        "live chat",
        "support team",
        "still waiting",
        "not resolved",
    ],
}

counts = {}

for intent, keywords in rules.items():
    mask = pd.Series(False, index=customer.index)

    for keyword in keywords:
        mask = mask | text.str.contains(
            keyword,
            regex=False,
            na=False
        )

    counts[intent] = int(mask.sum())

result = (
    pd.Series(counts, name="matched_messages")
    .sort_values(ascending=False)
)

print(result.to_string())