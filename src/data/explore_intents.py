from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

INPUT_PATH = Path("data/processed/amazon_corpus_clean.csv")

df = pd.read_csv(
    INPUT_PATH,
    usecols=["inbound", "clean_text"]
)

customer = df[df["inbound"] == True].copy()
customer["clean_text"] = customer["clean_text"].fillna("")

vectorizer = TfidfVectorizer(
    stop_words="english",
    min_df=30,
    max_features=5000,
    ngram_range=(2, 3)
)

matrix = vectorizer.fit_transform(customer["clean_text"])

scores = matrix.sum(axis=0).A1
terms = vectorizer.get_feature_names_out()

phrases = (
    pd.DataFrame(
        {
            "phrase": terms,
            "score": scores
        }
    )
    .sort_values("score", ascending=False)
    .head(150)
)

phrases.to_csv(
    "data/processed/top_customer_phrases.csv",
    index=False
)

print(phrases.to_string(index=False))