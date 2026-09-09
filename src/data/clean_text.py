import re
from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/amazon_corpus.csv")
OUTPUT_PATH = Path("data/processed/amazon_corpus_clean.csv")

df = pd.read_csv(INPUT_PATH)

def clean_text(text):
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\b\d{4,}\b", " ", text)
    text = text.replace("&amp;", "and")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

df.to_csv(OUTPUT_PATH, index=False)

print("Rows:", len(df))
print("Output:", OUTPUT_PATH)