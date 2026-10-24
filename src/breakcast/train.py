from .dataset import LABELS
from .metrics import evaluate
from .models import MODEL_NAMES, build, failure_scores, to_frame
from .splits import split_rows

MIN_PER_CLASS = 30
MIN_TRAIN_PROJECTS = 10


class InsufficientData(RuntimeError):
    pass


def labelled(rows):
    return [row for row in rows if row.outcome in LABELS]


def class_counts(rows):
    counts = {0: 0, 1: 0}
    for row in rows:
        counts[row.label] += 1
    return counts


def check_enough(parts, min_per_class=MIN_PER_CLASS, min_projects=MIN_TRAIN_PROJECTS):
    train, dev = parts["train"], parts["dev"]
    problems = []
    counts = class_counts(train)
    for label, name in ((0, "passes"), (1, "failures")):
        if counts[label] < min_per_class:
            problems.append("training has %d %s, needs at least %d" % (counts[label], name, min_per_class))
    projects = len({row.group for row in train})
    if projects < min_projects:
        problems.append("training has %d projects, needs at least %d" % (projects, min_projects))
    dev_counts = class_counts(dev)
    if not dev_counts[0] or not dev_counts[1]:
        problems.append("development has %d passes and %d failures, it needs both" % (dev_counts[0], dev_counts[1]))
    if problems:
        raise InsufficientData("not enough verified two-class data to train honestly: " + "; ".join(problems))


def score_part(model, rows):
    frame = to_frame([row.features for row in rows])
    labels = [row.label for row in rows]
    return evaluate(labels, failure_scores(model, frame), [row.group for row in rows])


def train_all(rows, manifest, min_per_class=MIN_PER_CLASS, min_projects=MIN_TRAIN_PROJECTS):
    parts = split_rows(labelled(rows), manifest)
    check_enough(parts, min_per_class, min_projects)
    frame = to_frame([row.features for row in parts["train"]])
    labels = [row.label for row in parts["train"]]
    models, report = {}, {}
    for name in MODEL_NAMES:
        model = build(name)
        model.fit(frame, labels)
        models[name] = model
        report[name] = {"train": score_part(model, parts["train"]), "dev": score_part(model, parts["dev"])}
    return models, report


def choose(report):
    ranked = sorted(report, key=lambda name: (report[name]["dev"]["pr_auc"] is not None, report[name]["dev"]["pr_auc"] or 0.0), reverse=True)
    return ranked[0]


def final_test(models, rows, manifest, chosen):
    parts = split_rows(labelled(rows), manifest)
    if not parts["test"]:
        raise InsufficientData("the test partition is empty")
    return {name: score_part(model, parts["test"]) for name, model in models.items() if name in (chosen, "majority", "major_jump_rule")}
