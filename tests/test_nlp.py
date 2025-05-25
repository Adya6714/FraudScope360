from modules.nlp.tfidf_logistic import NLPModule

def test_nlp_module_basic():
    texts = ["this is fraud", "all good here", "possible FRAUD spotted"]
    nlp = NLPModule(ngram_range=(1,2), C=1.0)
    # Should train without exception
    nlp.fit(texts)
    # Score on a new text returns a float between 0 and 1
    prob = nlp.score("fraudulent activity detected")
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0