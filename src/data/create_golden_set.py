from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/amazon_corpus_clean.csv")
OUTPUT_PATH = Path("data/golden/golden_set.csv")

df = pd.read_csv(INPUT_PATH)

customer = df[df["inbound"] == True].copy()

customer["text_length"] = customer["clean_text"].fillna("").str.len()

customer = customer[
    (customer["text_length"] >= 20)
    & (customer["text_length"] <= 500)
]

sample = customer.sample(
    n=200,
    random_state=42
)

golden = sample[
    [
        "conversation_id",
        "tweet_id",
        "text",
        "clean_text",
    ]
].copy()

golden["intent"] = ""
golden["labeler_notes"] = ""

golden.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Golden examples:", len(golden))
print("Output:", OUTPUT_PATH)