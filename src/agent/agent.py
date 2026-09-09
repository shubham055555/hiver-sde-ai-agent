import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

try:
    from agent.decision import make_decision
except ModuleNotFoundError:
    from decision import make_decision

from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=api_key)


GOLDEN_PATH = Path("data/golden/golden_set.csv")
RETRIEVAL_PATH = Path("data/processed/retrieval_pairs.csv")


golden_df = pd.read_csv(GOLDEN_PATH)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=10000
)

X = vectorizer.fit_transform(golden_df["text"].fillna(""))

classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

classifier.fit(
    X,
    golden_df["intent"]
)


retrieval_df = pd.read_csv(RETRIEVAL_PATH)

retrieval_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000
)

retrieval_matrix = retrieval_vectorizer.fit_transform(
    retrieval_df["customer_clean_text"].fillna("")
)


def predict_intent(message):
    message_vector = vectorizer.transform([message])
    return classifier.predict(message_vector)[0]


def retrieve_cases(message, top_k=3):
    message_vector = retrieval_vectorizer.transform([message])

    scores = cosine_similarity(
        message_vector,
        retrieval_matrix
    ).flatten()

    top_indices = scores.argsort()[-top_k:][::-1]

    results = retrieval_df.iloc[top_indices].copy()
    results["score"] = scores[top_indices]

    return results


def generate_reply(message, intent, cases):
    examples = []

    for _, case in cases.iterrows():
        examples.append(
            f"Customer: {case['customer_text']}\n"
            f"Amazon response: {case['amazon_reply']}"
        )

    historical_context = "\n\n".join(examples)

    prompt = f"""
You are a customer support agent for Amazon.

Customer message:
{message}

Predicted intent:
{intent}

Historical Amazon support examples:

{historical_context}

Write a concise and helpful reply to the current customer.

Use the historical examples only as grounding for response style and resolution approach.
Do not invent order details, refunds, delivery dates, policies, or actions.
Do not mention the historical examples.
Do not mention that you are an AI.
If information is missing, ask only for the information needed to investigate the issue.
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            return response.text.strip()

        except Exception:
            if attempt == 2:
                raise

            time.sleep(3)


def run_agent(message):
    intent = predict_intent(message)

    cases = retrieve_cases(message)

    if cases.empty:
        best_score = 0.0
    else:
        best_score = float(cases.iloc[0]["score"])

    decision_result = make_decision(
        intent,
        message,
        best_score
    )

    decision = decision_result["decision"]
    reason = decision_result["reason"]

    reply = None

    if decision == "auto_handle":
        reply = generate_reply(
            message,
            intent,
            cases
        )

    return {
        "message": message,
        "intent": intent,
        "decision": decision,
        "reason": reason,
        "retrieval_score": best_score,
        "reply": reply
    }


if __name__ == "__main__":
    message = input("Customer message: ")

    result = run_agent(message)

    print("\nIntent:", result["intent"])
    print("Decision:", result["decision"])
    print("Reason:", result["reason"])
    print("Retrieval score:", result["retrieval_score"])

    if result["reply"]:
        print("\nDraft reply:")
        print(result["reply"])