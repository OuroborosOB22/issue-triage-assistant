import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import FEATURE_SCHEMA_VERSION

OUTCOMES = ("OBSERVED_PASS", "OBSERVED_REGRESSION", "INCONCLUSIVE")
LABEL_SOURCES = ("bump_metadata", "sentry_evidence", "fixture")
LABELS = {"OBSERVED_PASS": 0, "OBSERVED_REGRESSION": 1}


class InvalidRow(ValueError):
    pass


@dataclass
class Row:
    row_id: str
    project: str
    package: str
    from_version: str
    to_version: str
    outcome: str
    label_source: str
    features: dict
    missing_features: list
    provenance: dict = field(default_factory=dict)
    feature_schema_version: str = FEATURE_SCHEMA_VERSION

    @property
    def group(self):
        return self.project.strip().lower()

    @property
    def label(self):
        return LABELS.get(self.outcome)

    def validate(self):
        if self.outcome not in OUTCOMES:
            raise InvalidRow("%s: unknown outcome %s" % (self.row_id, self.outcome))
        if self.label_source not in LABEL_SOURCES:
            raise InvalidRow("%s: unknown label source %s" % (self.row_id, self.label_source))
        if self.outcome == "OBSERVED_PASS" and self.label_source == "bump_metadata":
            raise InvalidRow("%s: BUMP records are failures and cannot be passes" % self.row_id)
        if not self.project or ":" not in self.package:
            raise InvalidRow("%s: project and a groupId:artifactId package are required" % self.row_id)
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise InvalidRow("%s: feature schema %s is not %s" % (self.row_id, self.feature_schema_version, FEATURE_SCHEMA_VERSION))
        unlisted = [name for name, value in self.features.items() if value is None and name not in self.missing_features]
        if unlisted:
            raise InvalidRow("%s: empty features not listed as missing: %s" % (self.row_id, unlisted))

    def key(self):
        return (self.group, self.package, self.from_version, self.to_version)


def write_rows(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in sorted(rows, key=lambda item: item.row_id):
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def read_rows(path):
    rows = []
    with open(path) as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = Row(**json.loads(line))
            except (TypeError, json.JSONDecodeError) as problem:
                raise InvalidRow("line %d: %s" % (number, problem)) from problem
            row.validate()
            rows.append(row)
    return rows


def deduplicate(rows):
    seen = {}
    for row in rows:
        previous = seen.get(row.key())
        if previous is None or (previous.label_source == "bump_metadata" and row.label_source == "sentry_evidence"):
            seen[row.key()] = row
    return list(seen.values())


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(rows):
    counts = {"rows": len(rows), "projects": len({row.group for row in rows}), "by_outcome": {}, "by_source": {}}
    for row in rows:
        counts["by_outcome"][row.outcome] = counts["by_outcome"].get(row.outcome, 0) + 1
        counts["by_source"][row.label_source] = counts["by_source"].get(row.label_source, 0) + 1
    return counts
