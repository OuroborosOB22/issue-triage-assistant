import json
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from .dataset import Row, file_digest
from .features import extract

CONTRACT_DIR = Path(__file__).resolve().parents[2] / "contracts" / "v1"
SCHEMA_NAMES = ["request", "evidence", "prediction_request", "prediction_response", "snapshot", "comparison", "worker_result"]
REGRESSION_CODES = {"test-regression", "new-failing-tests", "upgrade-build-failed", "upgrade-compile-failed", "upgrade-dependency-conflict",
                    "new-dependency-conflict", "tests-could-not-start"}


def registry(directory=CONTRACT_DIR):
    result = Registry()
    for name in SCHEMA_NAMES:
        contents = json.loads((Path(directory) / ("%s.schema.json" % name)).read_text())
        result = result.with_resource(contents["$id"], Resource.from_contents(contents))
    return result


def validator(name, directory=CONTRACT_DIR):
    schema = json.loads((Path(directory) / ("%s.schema.json" % name)).read_text())
    return Draft202012Validator(schema, registry=registry(directory))


def outcome_of(evidence):
    decision = evidence["verdict"]["decision"]
    codes = {finding["code"] for finding in evidence["verdict"]["findings"]}
    if decision == "ACCEPT":
        return "OBSERVED_PASS"
    if decision == "REJECT" and codes & REGRESSION_CODES:
        return "OBSERVED_REGRESSION"
    return "INCONCLUSIVE"


def row_from_evidence(evidence, project=None, path=None):
    change = evidence["request"]["change"]
    features, missing = extract(change["from_version"], change["to_version"])
    provenance = {"run_id": evidence["run_id"], "decision": evidence["verdict"]["decision"], "locally_verified": True}
    if path:
        provenance["evidence_sha256"] = file_digest(path)
    return Row(row_id="evidence-" + evidence["run_id"], project=project or evidence["request"]["project_id"], package=change["package"], from_version=change["from_version"],
               to_version=change["to_version"], outcome=outcome_of(evidence), label_source="sentry_evidence", features=features,
               missing_features=missing, provenance=provenance)


def load_evidence_rows(directory, project=None):
    check = validator("evidence")
    rows, rejected = [], []
    for path in sorted(Path(directory).rglob("evidence.json")):
        data = json.loads(path.read_text())
        problems = [error.message for error in check.iter_errors(data)]
        if problems:
            rejected.append({"file": str(path), "reason": problems[0]})
        elif data["request"]["change"]["ecosystem"] != "MAVEN":
            rejected.append({"file": str(path), "reason": "not a Maven run"})
        else:
            rows.append(row_from_evidence(data, project, path))
    return rows, rejected
