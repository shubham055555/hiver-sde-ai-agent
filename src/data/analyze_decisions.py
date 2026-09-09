import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/processed/agent_decisions.csv")


df = pd.read_csv(DATA_PATH)


summary = (
    df.groupby(["expected_intent", "decision"])
    .size()
    .unstack(fill_value=0)
)


print("Decision breakdown by expected intent:")
print(summary)


print("\nDecision percentage by expected intent:")

percentages = summary.div(
    summary.sum(axis=1),
    axis=0
) * 100

print(percentages.round(1))


print("\nAverage retrieval score by decision:")

print(
    df.groupby("decision")["retrieval_score"]
    .agg(["count", "mean", "min", "max"])
    .round(4)
)