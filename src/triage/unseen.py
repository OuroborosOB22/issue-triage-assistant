import json
import time
from pathlib import Path

from . import data, metrics, models, setfit_model, train

REPORT = "unseen_projects.json"
LOGREG_C = 30.0
CHECKPOINT = "sentence-transformers/all-MiniLM-L12-v2"


def score_fold(model_kind, fit_issues, held_out):
    if model_kind == "majority":
        model = models.majority()
        models.fit(model, fit_issues)
        return models.predict_scores(model, [issue.text for issue in held_out])
    if model_kind == "tfidf":
        model = models.tfidf_logreg(LOGREG_C)
        models.fit(model, fit_issues)
        return models.predict_scores(model, [issue.text for issue in held_out])
    model = setfit_model.fit(fit_issues, CHECKPOINT, iterations=10, epochs=1, device=score_fold.device)
    return setfit_model.predict_scores(model, [issue.text for issue in held_out])


score_fold.device = None


def run(raw_dir, reports_dir, kinds, device=None):
    score_fold.device = device
    prepared = train.prepare(raw_dir)
    train_clean = prepared["fit"] + prepared["dev"]
    test = prepared["test"]
    results = {}
    for held in sorted({issue.repo for issue in test}):
        others = [issue for issue in train_clean if issue.repo != held]
        fit_issues, _ = data.dev_split(others)
        held_out = [issue for issue in test if issue.repo == held]
        labels = [issue.label for issue in held_out]
        results[held] = {"fit_rows": len(fit_issues), "held_out_rows": len(held_out)}
        for kind in kinds:
            started = time.perf_counter()
            scores = score_fold(kind, fit_issues, held_out)
            report = metrics.evaluate(labels, scores)
            results[held][kind] = {"macro_f1": report["macro_f1"], "accuracy": report["accuracy"], "seconds": round(time.perf_counter() - started, 1)}
            print(held, kind, round(report["macro_f1"], 4), flush=True)
    summary = {kind: sum(results[repo][kind]["macro_f1"] for repo in results) / len(results) for kind in kinds}
    out = {"note": "Each project in turn is left out of training and its test issues are scored. This is a new study; the saved final test of the shipped model is not changed.",
           "models": kinds, "checkpoint": CHECKPOINT, "mean_macro_f1_on_unseen_project": summary, "projects": results}
    target = Path(reports_dir) / REPORT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out
