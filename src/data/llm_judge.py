import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai


INPUT_PATH = Path("data/processed/response_eval_results.csv")
OUTPUT_PATH = Path("data/processed/llm_judge_results.csv")


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=api_key)


def judge_response(row):
    prompt = f"""
You are evaluating an AI customer support response for Amazon.

Customer message:
{row["customer_message"]}

Expected intent:
{row["expected_intent"]}

Predicted intent:
{row["predicted_intent"]}

Draft response:
{row["draft_reply"]}

Evaluate the response using these five dimensions.

Groundedness:
How well the response follows the resolution approach supported by the historical examples.

Helpfulness:
Whether the response addresses the customer's issue and gives a useful next step.

Intent correctness:
Whether the response is appropriate for the customer's intent.

Unsupported claims:
Whether the response avoids inventing policies, refunds, dates, order details, promises, or actions.

Overall quality:
Whether the response is safe, grounded, helpful, concise, and suitable for automated support.

Give each dimension a score from 1 to 5.

Return only valid JSON with this structure:

{{
    "groundedness": 1,
    "helpfulness": 1,
    "intent_correctness": 1,
    "no_unsupported_claims": 1,
    "overall_quality": 1,
    "reason": "brief explanation"
}}
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            text = response.text.strip()

            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "").strip()

            return json.loads(text)

        except Exception as error:
            if attempt == 2:
                raise

            print(f"Judge request failed: {error}")
            print("Waiting before retry...")
            time.sleep(20 * (attempt + 1))


if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"{INPUT_PATH} does not exist. Generate response evaluation results first."
    )


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

    scores = judge_response(row)

    results.append(
        {
            "tweet_id": row["tweet_id"],
            "customer_message": row["customer_message"],
            "expected_intent": row["expected_intent"],
            "predicted_intent": row["predicted_intent"],
            "retrieval_score": row["retrieval_score"],
            "draft_reply": row["draft_reply"],
            "groundedness": scores["groundedness"],
            "helpfulness": scores["helpfulness"],
            "intent_correctness": scores["intent_correctness"],
            "no_unsupported_claims": scores["no_unsupported_claims"],
            "overall_quality": scores["overall_quality"],
            "judge_reason": scores["reason"]
        }
    )

    pd.DataFrame(results).to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Completed: {len(results)}/{len(df)}")


print(f"Judged examples: {len(results)}")
print(f"Saved to: {OUTPUT_PATH}")