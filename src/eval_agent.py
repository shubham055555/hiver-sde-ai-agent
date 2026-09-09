import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold


GOLDEN_PATH = Path("data/golden/golden_set.csv")


df = pd.read_csv(GOLDEN_PATH)
df = df.dropna(subset=["clean_text", "intent"])

X = df["clean_text"]
y = df["intent"]


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


all_predictions = []
all_actual = []


for train_index, test_index in cv.split(X, y):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=10000
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    all_predictions.extend(predictions)
    all_actual.extend(y_test)


accuracy = accuracy_score(
    all_actual,
    all_predictions
)

macro_f1 = f1_score(
    all_actual,
    all_predictions,
    average="macro"
)


print(f"Examples evaluated: {len(all_actual)}")
print(f"5-Fold Accuracy: {accuracy:.4f}")
print(f"5-Fold Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        all_actual,
        all_predictions,
        zero_division=0
    )
)


errors = df.iloc[
    [
        i for i in range(len(df))
        if all_actual[i] != all_predictions[i]
    ]
].copy()

errors["predicted_intent"] = [
    all_predictions[i]
    for i in range(len(df))
    if all_actual[i] != all_predictions[i]
]


print(f"\nIncorrect predictions: {len(errors)}")

print("\nSample errors:")

for _, row in errors.head(10).iterrows():
    print("\nCustomer:", row["text"])
    print("Expected:", row["intent"])
    print("Predicted:", row["predicted_intent"])