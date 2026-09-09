import os

import pandas as pd
from dotenv import load_dotenv
from google import genai
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path("data/processed/retrieval_pairs.csv")


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=api_key)


df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["customer_clean_text", "amazon_reply_clean"])


vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000
)

matrix = vectorizer.fit_transform(df["customer_clean_text"])


def retrieve_cases(customer_message, top_k=3):
    query_vector = vectorizer.transform([customer_message])
    scores = cosine_similarity(query_vector, matrix).flatten()

    indexes = scores.argsort()[-top_k:][::-1]

    results = df.iloc[indexes].copy()
    results["score"] = scores[indexes]

    return results


def generate_reply(customer_message, intent):
    cases = retrieve_cases(customer_message)

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
{customer_message}

Predicted intent:
{intent}

Here are historical Amazon customer-support examples from similar cases:

{historical_context}

Write a concise customer-support reply for the current customer.

Use the historical examples as grounding for the type of response and resolution style.
Do not invent order details, refunds, delivery dates, policies, or actions that are not supported by the context.
Do not mention the historical examples.
Do not mention that you are an AI.
If important information is missing, ask the customer for the information needed to investigate the issue.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text.strip(), cases


if __name__ == "__main__":
    customer_message = input("Customer message: ").strip()
    intent = input("Intent: ").strip()

    reply, cases = generate_reply(customer_message, intent)

    print("\nRetrieved cases:")

    for _, case in cases.iterrows():
        print("\nScore:", round(case["score"], 4))
        print("Customer:", case["customer_text"])
        print("Amazon:", case["amazon_reply"])

    print("\nGenerated reply:")
    print(reply)