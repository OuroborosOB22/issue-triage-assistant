import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import ALL, CATEGORICAL, NUMERIC, VOCABULARY

SEED = 7
MODEL_NAMES = ("majority", "major_jump_rule", "logistic_regression", "gradient_boosting")


def to_frame(feature_dicts):
    frame = pd.DataFrame(list(feature_dicts), columns=ALL)
    for name in NUMERIC:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    for name in CATEGORICAL:
        frame[name] = frame[name].astype("object").where(frame[name].notna(), "missing")
    return frame


class MajorJumpRule(BaseEstimator, ClassifierMixin):
    def fit(self, frame, labels):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, frame):
        risky = (frame["change_kind"] == "major").to_numpy().astype(float)
        return np.column_stack([1 - risky, risky])

    def predict(self, frame):
        return (self.predict_proba(frame)[:, 1] >= 0.5).astype(int)


def preprocessing(scale):
    numeric_steps = [("fill", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    categories = [VOCABULARY[name] + ["missing"] for name in CATEGORICAL]
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC),
        ("categorical", OneHotEncoder(categories=categories, handle_unknown="ignore"), CATEGORICAL),
    ])


def build(name):
    if name == "majority":
        return DummyClassifier(strategy="prior")
    if name == "major_jump_rule":
        return MajorJumpRule()
    if name == "logistic_regression":
        return Pipeline([("prepare", preprocessing(True)), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))])
    if name == "gradient_boosting":
        return Pipeline([("prepare", preprocessing(False)), ("model", HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.1, random_state=SEED))])
    raise ValueError("unknown model %s" % name)


def failure_scores(model, frame):
    probabilities = model.predict_proba(frame)
    classes = list(model.classes_)
    return probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(frame))


def contributions(model, features):
    if not isinstance(model, Pipeline) or not isinstance(model.named_steps["model"], LogisticRegression):
        return {}
    frame = to_frame([features])
    prepare, logistic = model.named_steps["prepare"], model.named_steps["model"]
    names = prepare.get_feature_names_out()
    values = prepare.transform(frame)
    row = values.toarray()[0] if hasattr(values, "toarray") else np.asarray(values)[0]
    result = {}
    for name, value, weight in zip(names, row, logistic.coef_[0]):
        base = name.split("__", 1)[1]
        for feature in ALL:
            if base == feature or base.startswith(feature + "_"):
                result[feature] = result.get(feature, 0.0) + float(value * weight)
                break
    return result
