import pandas as pd
from pathlib import Path


GOLDEN_PATH = Path("data/golden/golden_set.csv")
OUTPUT_PATH = Path("data/golden/response_eval_set.csv")


df = pd.read_csv(GOLDEN_PATH)
df = df.dropna(subset=["clean_text", "intent"])

response_intents = [
    "account_access",
    "delivery_issue",
    "order_management",
    "payment_billing",
    "prime_subscription",
    "product_information",
    "product_technical",
    "returns_refunds",
    "seller_review",
    "other_non_support"
]

selected_parts = []

for intent in response_intents:
    group = df[df["intent"] == intent]

    if len(group) >= 2:
        selected_parts.append(
            group.sample(
                n=2,
                random_state=42
            )
        )
    elif len(group) == 1:
        selected_parts.append(group)

selected = pd.concat(
    selected_parts,
    ignore_index=True
)

selected["human_grounded"] = ""
selected["human_helpful"] = ""
selected["human_correct_intent"] = ""
selected["human_no_unsupported_claims"] = ""
selected["human_overall"] = ""
selected["judge_notes"] = ""

selected.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"Evaluation examples: {len(selected)}")
print(f"Saved to: {OUTPUT_PATH}")

print("\nIntent distribution:")
print(selected["intent"].value_counts())