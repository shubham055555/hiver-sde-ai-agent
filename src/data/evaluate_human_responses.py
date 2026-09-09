import pandas as pd
from pathlib import Path


INPUT_PATH = Path("data/processed/response_eval_results.csv")
OUTPUT_PATH = Path("data/golden/response_human_eval.csv")


if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"{INPUT_PATH} does not exist. Generate response evaluation results first."
    )


df = pd.read_csv(INPUT_PATH)

columns = [
    "tweet_id",
    "customer_message",
    "expected_intent",
    "predicted_intent",
    "retrieval_score",
    "draft_reply",
    "human_grounded",
    "human_helpful",
    "human_correct_intent",
    "human_no_unsupported_claims",
    "human_overall",
    "judge_notes"
]

for column in columns:
    if column not in df.columns:
        df[column] = ""

df[columns].to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"Examples ready for human evaluation: {len(df)}")
print(f"Saved to: {OUTPUT_PATH}")