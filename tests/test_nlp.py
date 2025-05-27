import pandas as pd
from modules.nlp.tfidf_logistic import NLPModule

def test_nlp_module_on_real_data():
    # 1) Load your labeled transactions
    df = pd.read_csv("data/transactions_labeled.csv")
    assert "merchant" in df.columns and "is_fraud" in df.columns

    # 2) Build training texts by appending "fraud" or "legit"
    sample = df.head(200)
    texts = [
        f"{m} fraud" if y == 1 else f"{m} legit"
        for m, y in zip(sample["merchant"], sample["is_fraud"])
    ]

    # 3) Initialize & fit
    nlp = NLPModule(ngram_range=(1,2), C=1.0)
    nlp.fit(texts)

    # 4) Take a fresh example from later in the CSV
    test_row = df.iloc[201]
    test_text = f"{test_row['merchant']} fraud" if test_row["is_fraud"] == 1 else f"{test_row['merchant']} legit"

    # 5) Score it
    prob = nlp.score(test_text)

    # 6) Verify
    assert isinstance(prob, float), "Expected a float probability"
    assert 0.0 <= prob <= 1.0, "Probability must be between 0 and 1"