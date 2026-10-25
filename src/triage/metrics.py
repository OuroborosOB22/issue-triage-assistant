import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from . import LABELS


def top(scores):
    label = max(scores, key=scores.get)
    return label, float(scores[label])


def evaluate(true_labels, scores, repos=None):
    predicted = [top(row)[0] for row in scores]
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, predicted, labels=list(LABELS), zero_division=0)
    report = {
        "rows": len(true_labels),
        "macro_f1": float(f1_score(true_labels, predicted, labels=list(LABELS), average="macro", zero_division=0)),
        "accuracy": float(np.mean([a == b for a, b in zip(true_labels, predicted)])),
        "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
                      for i, label in enumerate(LABELS)},
        "confusion_matrix": {"labels": list(LABELS), "matrix": confusion_matrix(true_labels, predicted, labels=list(LABELS)).tolist()},
    }
    if repos is not None:
        report["per_repo_macro_f1"] = {}
        for repo in sorted(set(repos)):
            picks = [i for i, name in enumerate(repos) if name == repo]
            report["per_repo_macro_f1"][repo] = float(f1_score([true_labels[i] for i in picks], [predicted[i] for i in picks],
                                                               labels=list(LABELS), average="macro", zero_division=0))
        report["cross_repo_mean_f1"] = float(np.mean(list(report["per_repo_macro_f1"].values())))
    return report


def review_threshold(true_labels, scores, target_accuracy=0.9):
    pairs = sorted(((top(row)[1], top(row)[0] == truth) for row, truth in zip(scores, true_labels)), reverse=True)
    best = None
    correct = 0
    seen = 0
    index = 0
    while index < len(pairs):
        confidence = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == confidence:
            correct += pairs[index][1]
            seen += 1
            index += 1
        if correct / seen >= target_accuracy:
            best = confidence
    return best


def coverage(true_labels, scores, threshold):
    if threshold is None:
        return {"threshold": None, "accepted": 0.0, "accuracy_on_accepted": None}
    kept = [(top(row)[0] == truth) for row, truth in zip(scores, true_labels) if top(row)[1] >= threshold]
    return {"threshold": threshold, "accepted": len(kept) / len(true_labels), "accuracy_on_accepted": (sum(kept) / len(kept)) if kept else None}
