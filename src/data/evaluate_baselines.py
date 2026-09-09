import pandas as pd
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


INPUT_PATH = Path("data/golden/golden_set.csv")
OUTPUT_PATH = Path("data/processed/baseline_results.csv")


df = pd.read_csv(INPUT_PATH)
df = df.dropna(subset=["text", "intent"]).reset_index(drop=True)

X_text = df["text"]
y = df["intent"]


majority_intent = y.value_counts().idxmax()
majority_predictions = [majority_intent] * len(y)

majority_accuracy = accuracy_score(
    y,
    majority_predictions
)

majority_macro_f1 = f1_score(
    y,
    majority_predictions,
    average="macro",
    zero_division=0
)


vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=10000
)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

tfidf_predictions = []
tfidf_actual = []

for train_idx, test_idx in skf.split(X_text, y):
    train_text = X_text.iloc[train_idx]
    test_text = X_text.iloc[test_idx]

    train_labels = y.iloc[train_idx]
    test_labels = y.iloc[test_idx]

    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    classifier.fit(
        X_train,
        train_labels
    )

    predictions = classifier.predict(X_test)

    tfidf_predictions.extend(predictions)
    tfidf_actual.extend(test_labels)


tfidf_accuracy = accuracy_score(
    tfidf_actual,
    tfidf_predictions
)

tfidf_macro_f1 = f1_score(
    tfidf_actual,
    tfidf_predictions,
    average="macro",
    zero_division=0
)


results = pd.DataFrame(
    [
        {
            "baseline": "majority_class",
            "accuracy": majority_accuracy,
            "macro_f1": majority_macro_f1
        },
        {
            "baseline": "tfidf_logistic_regression",
            "accuracy": tfidf_accuracy,
            "macro_f1": tfidf_macro_f1
        }
    ]
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results.to_csv(
    OUTPUT_PATH,
    index=False
)

print(results.to_string(index=False))
print(f"\nSaved to: {OUTPUT_PATH}")