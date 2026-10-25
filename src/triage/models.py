from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

LOGREG_C = (0.5, 1.0, 3.0, 10.0, 30.0)


def majority():
    return Pipeline([("text", TfidfVectorizer(max_features=10)), ("model", DummyClassifier(strategy="most_frequent"))])


def tfidf_logreg(c):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60000)
    return Pipeline([("text", vectorizer), ("model", LogisticRegression(C=c, max_iter=3000))])


def fit(model, issues):
    model.fit([issue.text for issue in issues], [issue.label for issue in issues])
    return model


def predict_scores(model, texts):
    probabilities = model.predict_proba(texts)
    return [dict(zip(model.classes_, row)) for row in probabilities]
