import pandas as pd
from pathlib import Path

from agent.agent import predict_intent, retrieve_cases, generate_reply


INPUT_PATH = Path("data/golden/response_eval_set.csv")
OUTPUT_PATH = Path("data/processed/response_eval_results.csv")


df = pd.read_csv(INPUT_PATH)

results = []

for _, row in df.iterrows():
    message = row["text"]

    intent = predict_intent(message)
    cases = retrieve_cases(message, top_k=3)

    if not cases.empty:
        retrieval_score = cases.iloc[0]["score"]
    else:
        retrieval_score = 0.0

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


results_df = pd.DataFrame(results)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"Evaluated examples: {len(results_df)}")
print(f"Saved to: {OUTPUT_PATH}")