import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path("data/processed/retrieval_pairs.csv")

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["customer_clean_text", "amazon_reply_clean"])

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000
)

matrix = vectorizer.fit_transform(df["customer_clean_text"])


def find_similar(query, top_k=5):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).flatten()
    indexes = scores.argsort()[-top_k:][::-1]

    results = df.iloc[indexes].copy()
    results["score"] = scores[indexes]

    return results[
        ["customer_text", "amazon_reply", "score"]
    ]


if __name__ == "__main__":
    query = input("Customer message: ").strip()

    results = find_similar(query)

    for i, row in results.iterrows():
        print("\nScore:", round(row["score"], 4))
        print("Customer:", row["customer_text"])
        print("Amazon:", row["amazon_reply"])