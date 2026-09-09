import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


DATA_PATH = Path("data/golden/golden_set.csv")


df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["clean_text", "intent"])

X = df["clean_text"]
y = df["intent"]


pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=10000
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


predictions = cross_val_predict(
    pipeline,
    X,
    y,
    cv=cv
)


accuracy = accuracy_score(y, predictions)
macro_f1 = f1_score(y, predictions, average="macro")


print(f"5-Fold Accuracy: {accuracy:.4f}")
print(f"5-Fold Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y,
        predictions,
        zero_division=0
    )
)

labels = sorted(y.unique())

matrix = confusion_matrix(
    y,
    predictions,
    labels=labels
)

print("\nConfusion Matrix:")
print(pd.DataFrame(
    matrix,
    index=labels,
    columns=labels
))