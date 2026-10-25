import csv
import hashlib
import random
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import LABELS

COMMIT = "2927bc67eb42db8affd16eaf3e5a6d74f3063961"
URL = "https://raw.githubusercontent.com/nlbse2024/issue-report-classification/%s/data/issues_%s.csv"
SHA256 = {
    "train": "18dc42a30aa33dccadb723ad3baeb164d38bff521496f985ca2791c26b8939f5",
    "test": "4f7d8619d4e5adbea126e548fd8c214449288f3a93bb3bc130c54cd307af7e85",
}
MAX_TEXT = 6000
DEV_FRACTION = 0.2
SPLIT_SEED = 7

csv.field_size_limit(sys.maxsize)


class BadData(ValueError):
    pass


@dataclass(frozen=True)
class Issue:
    repo: str
    label: str
    title: str
    body: str

    @property
    def text(self):
        return join_text(self.title, self.body)


def join_text(title, body):
    return ("%s\n\n%s" % (title.strip(), (body or "").strip()))[:MAX_TEXT]


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in SHA256.items():
        target = directory / ("issues_%s.csv" % name)
        if not target.is_file() or sha256_of(target) != expected:
            with urllib.request.urlopen(URL % (COMMIT, name), timeout=60) as response:
                target.write_bytes(response.read())
        if sha256_of(target) != expected:
            raise BadData("%s does not match the pinned file" % target.name)
    return {name: str(directory / ("issues_%s.csv" % name)) for name in SHA256}


def read(path):
    issues = []
    with open(path, encoding="utf8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["label"] not in LABELS:
                raise BadData("unknown label %r" % row["label"])
            issues.append(Issue(row["repo"], row["label"], row["title"] or "", row["body"] or ""))
    return issues


def key(issue):
    return (issue.title.strip(), issue.body.strip())


def clean_train(train, test):
    held_out = {key(issue) for issue in test}
    seen, kept, dropped = set(), [], {"duplicate": 0, "in_test": 0}
    for issue in train:
        if key(issue) in held_out:
            dropped["in_test"] += 1
        elif key(issue) in seen:
            dropped["duplicate"] += 1
        else:
            seen.add(key(issue))
            kept.append(issue)
    return kept, dropped


def dev_split(train, seed=SPLIT_SEED, fraction=DEV_FRACTION):
    groups = {}
    for issue in train:
        groups.setdefault((issue.repo, issue.label), []).append(issue)
    rng = random.Random(seed)
    fit, dev = [], []
    for name in sorted(groups):
        members = sorted(groups[name], key=lambda issue: hashlib.sha256(issue.text.encode()).hexdigest())
        rng.shuffle(members)
        cut = max(1, round(len(members) * fraction))
        dev += members[:cut]
        fit += members[cut:]
    return fit, dev


def inventory(issues):
    counts = {}
    for issue in issues:
        counts.setdefault(issue.repo, {label: 0 for label in LABELS})[issue.label] += 1
    return {"rows": len(issues), "by_repo": counts, "empty_body": sum(not issue.body.strip() for issue in issues)}
