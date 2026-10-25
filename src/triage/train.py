import json
import platform
import time
from pathlib import Path

import numpy as np

from . import data, metrics, models

REPORT_FINAL = "final_test.json"


def dev_report(model, issues):
    scores = models.predict_scores(model, [issue.text for issue in issues])
    return scores, metrics.evaluate([issue.label for issue in issues], scores, [issue.repo for issue in issues])


def compare_baselines(fit_issues, dev_issues):
    results = {}
    candidates = {"majority": models.majority()}
    for c in models.LOGREG_C:
        candidates["tfidf_logreg_C%s" % c] = models.tfidf_logreg(c)
    for name, model in candidates.items():
        started = time.perf_counter()
        models.fit(model, fit_issues)
        scores, report = dev_report(model, dev_issues)
        report["train_seconds"] = round(time.perf_counter() - started, 2)
        results[name] = {"model": model, "scores": scores, "dev": report}
    return results


SETFIT_MARGIN = 0.02


def best_name(results):
    best = max(results, key=lambda name: results[name]["dev"]["macro_f1"])
    plain = [name for name in results if not name.startswith("setfit")]
    if best.startswith("setfit") and plain:
        best_plain = max(plain, key=lambda name: results[name]["dev"]["macro_f1"])
        if results[best]["dev"]["macro_f1"] - results[best_plain]["dev"]["macro_f1"] <= SETFIT_MARGIN:
            return best_plain
    return best


def compare_setfit(fit_issues, dev_issues, checkpoints, iterations, epochs, device=None):
    from . import setfit_model
    results = {}
    for checkpoint in checkpoints:
        started = time.perf_counter()
        model = setfit_model.fit(fit_issues, checkpoint, iterations=iterations, epochs=epochs, device=device)
        seconds = round(time.perf_counter() - started, 1)
        scores = setfit_model.predict_scores(model, [issue.text for issue in dev_issues])
        report = metrics.evaluate([issue.label for issue in dev_issues], scores, [issue.repo for issue in dev_issues])
        report.update({"train_seconds": seconds, "checkpoint": checkpoint, "iterations": iterations, "epochs": epochs, "device": device or "cpu"})
        results["setfit_" + checkpoint.split("/")[-1]] = {"model": model, "scores": scores, "dev": report}
    return results


def prepare(raw_dir):
    paths = data.fetch(raw_dir)
    train_all, test = data.read(paths["train"]), data.read(paths["test"])
    train_clean, dropped = data.clean_train(train_all, test)
    fit_issues, dev_issues = data.dev_split(train_clean)
    return {"fit": fit_issues, "dev": dev_issues, "test": test, "dropped": dropped, "paths": paths}


def environment():
    return {"python": platform.python_version(), "machine": platform.machine(), "platform": platform.platform(), "processor": platform.processor()}


def final_test(predictor, manifest, manifest_sha256, test_issues, out_dir):
    target = Path(out_dir) / REPORT_FINAL
    if target.exists():
        raise FileExistsError("the held-out test was already scored (%s); delete the report only to start a new study" % target)
    threshold = manifest["review_threshold"]
    scores = predictor.scores([issue.text for issue in test_issues])
    report = metrics.evaluate([issue.label for issue in test_issues], scores, [issue.repo for issue in test_issues])
    report["review"] = metrics.coverage([issue.label for issue in test_issues], scores, threshold)
    wrong = sorted(((metrics.top(row)[1], issue, metrics.top(row)[0]) for row, issue in zip(scores, test_issues) if metrics.top(row)[0] != issue.label),
                   key=lambda item: -item[0])[:5]
    report["confident_mistakes"] = [{"repo": issue.repo, "title": issue.title[:140], "true": issue.label, "predicted": label, "score": round(score, 3)}
                                    for score, issue, label in wrong]
    latencies = []
    for issue in test_issues[:200]:
        started = time.perf_counter()
        predictor.scores([issue.text])
        latencies.append((time.perf_counter() - started) * 1000)
    report["latency_ms_single"] = {"median": float(np.median(latencies)), "p95": float(np.percentile(latencies, 95)), "samples": len(latencies)}
    report["environment"] = environment()
    report["model_version"] = manifest["model_version"]
    report["manifest_sha256"] = manifest_sha256
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return report
