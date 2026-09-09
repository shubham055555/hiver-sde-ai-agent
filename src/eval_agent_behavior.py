import pandas as pd
from pathlib import Path

from agent.agent import run_agent


GOLDEN_PATH = Path("data/golden/golden_set.csv")
OUTPUT_PATH = Path("data/processed/agent_behavior_results.csv")


df = pd.read_csv(GOLDEN_PATH)
df = df.dropna(subset=["clean_text", "intent"])


results = []

for index, row in df.iterrows():
    result = run_agent(row["clean_text"])

    results.append({
        "conversation_id": row["conversation_id"],
        "tweet_id": row["tweet_id"],
        "customer_text": row["text"],
        "expected_intent": row["intent"],
        "predicted_intent": result["intent"],
        "decision": result["decision"],
        "decision_reason": result["reason"],
        "retrieval_score": result["retrieval_score"],
        "draft_reply": result["reply"]
    })

    print(f"Processed {index + 1}/{len(df)}")


results_df = pd.DataFrame(results)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)

print("\nEvaluation complete")
print(f"Examples: {len(results_df)}")
print(f"Auto-handle: {(results_df['decision'] == 'auto_handle').sum()}")
print(f"Human escalation: {(results_df['decision'] == 'human_escalation').sum()}")
print(f"Saved to: {OUTPUT_PATH}")