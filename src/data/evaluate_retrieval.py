import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity


GOLDEN_PATH = Path("data/golden/golden_set.csv")
PAIRS_PATH = Path("data/processed/retrieval_pairs.csv")


golden = pd.read_csv(GOLDEN_PATH)
golden = golden.dropna(subset=["clean_text", "intent"])


classifier_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=10000
)

X = classifier_vectorizer.fit_transform(golden["clean_text"])

classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

classifier.fit(X, golden["intent"])


pairs = pd.read_csv(PAIRS_PATH)
pairs = pairs.dropna(subset=["customer_clean_text"])


pairs["retrieved_intent"] = classifier.predict(
    classifier_vectorizer.transform(pairs["customer_clean_text"])
)


retrieval_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000
)

retrieval_matrix = retrieval_vectorizer.fit_transform(
    pairs["customer_clean_text"]
)


top1_correct = 0
top3_correct = 0

total = len(golden)


for _, row in golden.iterrows():
    query_vector = retrieval_vectorizer.transform(
        [row["clean_text"]]
    )

    scores = cosine_similarity(
        query_vector,
        retrieval_matrix
    ).flatten()

    indexes = scores.argsort()[-3:][::-1]

    retrieved_intents = pairs.iloc[indexes]["retrieved_intent"].tolist()

    if retrieved_intents[0] == row["intent"]:
        top1_correct += 1

    if row["intent"] in retrieved_intents:
        top3_correct += 1


top1_accuracy = top1_correct / total
top3_accuracy = top3_correct / total


print(f"Golden examples: {total}")
print(f"Top-1 intent agreement: {top1_accuracy:.4f}")
print(f"Top-3 intent agreement: {top3_accuracy:.4f}")