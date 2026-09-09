import pandas as pd
from pathlib import Path

from agent.agent import predict_intent, retrieve_cases, generate_reply


INPUT_PATH = Path("data/golden/response_eval_set.csv")
OUTPUT_PATH = Path("data/processed/response_eval_results.csv")


df = pd.read_csv(INPUT_PATH)

results = []

if OUTPUT_PATH.exists():
    existing_df = pd.read_csv(OUTPUT_PATH)
    results = existing_df.to_dict("records")
    completed_ids = set(existing_df["tweet_id"].astype(str))
else:
    completed_ids = set()


for _, row in df.iterrows():
    tweet_id = str(row["tweet_id"])

    if tweet_id in completed_ids:
        continue

    message = row["text"]

    intent = predict_intent(message)
    cases = retrieve_cases(message, top_k=3)

    if cases.empty:
        retrieval_score = 0.0
    else:
        retrieval_score = float(cases.iloc[0]["score"])

    reply = generate_reply(message, intent, cases)

    results.append(
        {
            "tweet_id": row["tweet_id"],
            "customer_message": message,
            "expected_intent": row["intent"],
            "predicted_intent": intent,
            "retrieval_score": retrieval_score,
            "draft_reply": reply
        }
    )

    pd.DataFrame(results).to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Completed: {len(results)}/{len(df)}")


print(f"Evaluated examples: {len(results)}")
print(f"Saved to: {OUTPUT_PATH}")