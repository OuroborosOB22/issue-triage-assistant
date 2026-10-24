import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

BOOTSTRAPS = 200
CALIBRATION_BINS = 5


def undefined(reason):
    return {"value": None, "reason": reason}


def pr_auc(labels, scores):
    if len(set(labels)) < 2:
        return None
    return float(average_precision_score(labels, scores))


def roc_auc(labels, scores):
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def group_interval(labels, scores, groups, seed=0):
    names = sorted(set(groups))
    if len(names) < 5 or len(set(labels)) < 2:
        return None
    rng = np.random.default_rng(seed)
    labels, scores, groups = np.asarray(labels), np.asarray(scores), np.asarray(groups)
    values = []
    for _ in range(BOOTSTRAPS):
        picked = rng.choice(names, size=len(names), replace=True)
        mask = np.concatenate([np.flatnonzero(groups == name) for name in picked])
        if len(set(labels[mask])) == 2:
            values.append(average_precision_score(labels[mask], scores[mask]))
    if len(values) < BOOTSTRAPS // 2:
        return None
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def calibration_bins(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    edges = np.linspace(0, 1, CALIBRATION_BINS + 1)
    bins = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (scores >= low) & (scores <= high if high == 1.0 else scores < high)
        if mask.any():
            bins.append({"range": [float(low), float(high)], "count": int(mask.sum()), "mean_score": float(scores[mask].mean()),
                         "observed_failure_rate": float(labels[mask].mean())})
    return bins


def evaluate(labels, scores, groups, threshold=0.5):
    labels = list(labels)
    scores = [float(value) for value in scores]
    predicted = [int(value >= threshold) for value in scores]
    matrix = confusion_matrix(labels, predicted, labels=[0, 1]).tolist()
    (tn, fp), (fn, tp) = matrix
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "rows": len(labels),
        "failures": int(sum(labels)),
        "projects": len(set(groups)),
        "threshold": threshold,
        "pr_auc": pr_auc(labels, scores),
        "pr_auc_interval_95": group_interval(labels, scores, groups),
        "roc_auc": roc_auc(labels, scores),
        "precision": precision,
        "failure_recall": recall,
        "confusion_matrix": {"labels": [0, 1], "matrix": matrix},
        "brier": float(brier_score_loss(labels, scores)) if labels else None,
        "calibration": calibration_bins(labels, scores),
        "undefined_note": None if len(set(labels)) == 2 else "only one class present, ranking metrics are undefined",
    }
