import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/processed/agent_decisions.csv")


df = pd.read_csv(DATA_PATH)


intent_errors = df[
    df["expected_intent"] != df["predicted_intent"]
].copy()


print(f"Total examples: {len(df)}")
print(f"Intent errors: {len(intent_errors)}")

print("\nIntent error pairs:")

print(
    intent_errors
    .groupby(["expected_intent", "predicted_intent"])
    .size()
    .sort_values(ascending=False)
)


print("\nPotential false auto-handles:")

false_auto = df[
    (df["decision"] == "auto_handle")
    & (
        df["expected_intent"].isin(
            ["support_escalation", "other_non_support"]
        )
    )
]

print(f"Count: {len(false_auto)}")

for _, row in false_auto.head(10).iterrows():
    print("\nCustomer:", row["customer_text"])
    print("Expected intent:", row["expected_intent"])
    print("Predicted intent:", row["predicted_intent"])
    print("Decision:", row["decision"])
    print("Reason:", row["decision_reason"])
    print("Retrieval score:", round(row["retrieval_score"], 4))


print("\nPotential unnecessary escalations:")

unnecessary = df[
    (df["decision"] == "human_escalation")
    & (~df["expected_intent"].isin(
        ["support_escalation", "other_non_support"]
    ))
]

print(f"Count: {len(unnecessary)}")

for _, row in unnecessary.head(10).iterrows():
    print("\nCustomer:", row["customer_text"])
    print("Expected intent:", row["expected_intent"])
    print("Predicted intent:", row["predicted_intent"])
    print("Decision:", row["decision"])
    print("Reason:", row["decision_reason"])
    print("Retrieval score:", round(row["retrieval_score"], 4))