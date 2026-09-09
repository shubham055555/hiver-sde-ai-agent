import pandas as pd
from pathlib import Path

from agent.agent import predict_intent, retrieve_cases, make_decision


GOLDEN_PATH = Path("data/golden/golden_set.csv")
OUTPUT_PATH = Path("data/processed/agent_decisions.csv")


df = pd.read_csv(GOLDEN_PATH)
df = df.dropna(subset=["clean_text", "intent"])


results = []

for _, row in df.iterrows():
    message = row["clean_text"]

    intent = predict_intent(message)

    cases = retrieve_cases(message, top_k=3)

    score = float(cases.iloc[0]["score"])

    decision, reason = make_decision(
        intent,
        message,
        score
    )

    results.append({
        "conversation_id": row["conversation_id"],
        "tweet_id": row["tweet_id"],
        "customer_text": row["text"],
        "expected_intent": row["intent"],
        "predicted_intent": intent,
        "decision": decision,
        "decision_reason": reason,
        "retrieval_score": score
    })


results_df = pd.DataFrame(results)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)


print(f"Examples evaluated: {len(results_df)}")
print(
    f"Auto-handle: "
    f"{(results_df['decision'] == 'auto_handle').sum()}"
)
print(
    f"Human escalation: "
    f"{(results_df['decision'] == 'human_escalation').sum()}"
)
print(f"Saved to: {OUTPUT_PATH}")