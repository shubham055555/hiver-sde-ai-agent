import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/golden/golden_set.csv")
OUTPUT_PATH = Path("data/golden/golden_set.csv")

INTENTS = [
    "delivery_issue",
    "order_management",
    "returns_refunds",
    "payment_billing",
    "prime_subscription",
    "account_access",
    "product_technical",
    "product_information",
    "seller_review",
    "support_escalation",
    "other_non_support",
]

df = pd.read_csv(INPUT_PATH)

df["intent"] = df["intent"].fillna("").astype(str)
df["labeler_notes"] = df["labeler_notes"].fillna("").astype(str)

for i in range(len(df)):
    row = df.iloc[i]

    if row["intent"].strip():
        continue

    print("\n" + "=" * 80)
    print(f"Example {i + 1}/{len(df)}")
    print("=" * 80)
    print(row["text"])

    print("\nChoose intent:")

    for number, intent in enumerate(INTENTS, 1):
        print(f"{number}. {intent}")

    while True:
        choice = input("\nEnter number: ").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(INTENTS):
            intent = INTENTS[int(choice) - 1]
            break

        print("Invalid choice. Enter a number from 1 to 11.")

    note = input("Note (optional): ").strip()

    df.at[i, "intent"] = intent
    df.at[i, "labeler_notes"] = note

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {intent}")

print("\nGolden set labeling complete.")
print(f"Output: {OUTPUT_PATH}")