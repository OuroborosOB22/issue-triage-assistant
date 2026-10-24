import random

from .dataset import Row
from .features import extract

PACKAGES = ["org.example:alpha", "org.example:beta", "com.sample:gamma", "io.demo:delta", "org.test:epsilon"]


def fixture_rows(projects=24, per_project=6, seed=11):
    rng = random.Random(seed)
    rows = []
    for number in range(projects):
        for step in range(per_project):
            major = rng.randint(1, 4)
            kind = rng.choice(["major", "minor", "patch"])
            before = "%d.%d.%d" % (major, rng.randint(0, 9), rng.randint(0, 9))
            after = {"major": "%d.0.0" % (major + 1), "minor": "%d.%d.0" % (major, 10 + rng.randint(0, 5)), "patch": "%d.9.%d" % (major, 10 + rng.randint(0, 5))}[kind]
            chance = {"major": 0.7, "minor": 0.3, "patch": 0.1}[kind]
            failed = rng.random() < chance
            features, missing = extract(before, after, rng.choice(["compile", "test", "runtime"]), "dependencies", "JAR")
            rows.append(Row(row_id="fixture-%d-%d" % (number, step), project="org%d/project%d" % (number % 6, number), package=rng.choice(PACKAGES),
                            from_version=before, to_version=after, outcome="OBSERVED_REGRESSION" if failed else "OBSERVED_PASS", label_source="fixture",
                            features=features, missing_features=missing, provenance={"fixture": True, "note": "made up for tests, not a research result"}))
    return rows
