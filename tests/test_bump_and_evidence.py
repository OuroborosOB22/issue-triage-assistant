import json
import subprocess

import pytest

from breakcast import bump, evidence


def test_bump_record_becomes_an_unverified_regression_row(bump_record):
    row = bump.row_from_record(bump_record)
    assert row.outcome == "OBSERVED_REGRESSION"
    assert row.label_source == "bump_metadata"
    assert row.provenance["locally_verified"] is False
    assert row.package == "org.apache.mina:mina-core"
    assert row.group == "example/project"
    assert "failureCategory" not in json.dumps(row.features)


def test_bump_failure_category_never_reaches_the_features(bump_record):
    row = bump.row_from_record(bump_record)
    assert "COMPILATION_FAILURE" not in json.dumps(row.features)


def test_load_rows_skips_broken_files_and_says_why(tmp_path, bump_record):
    (tmp_path / "good.json").write_text(json.dumps(bump_record))
    (tmp_path / "bad.json").write_text("{\"project\": \"x\"}")
    rows, skipped = bump.load_rows(tmp_path)
    assert len(rows) == 1
    assert skipped[0]["file"] == "bad.json"


def test_fetch_benchmark_copies_only_the_metadata(tmp_path, bump_record):
    origin = tmp_path / "origin"
    (origin / "data" / "benchmark").mkdir(parents=True)
    (origin / "data" / "benchmark" / "a.json").write_text(json.dumps(bump_record))
    (origin / "other.txt").write_text("not data")
    for command in (["init", "-q", "-b", "main"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"]):
        subprocess.run(["git", *command], cwd=origin, check=True, capture_output=True)
    result = bump.fetch_benchmark(tmp_path / "out", repository="file://" + str(origin))
    assert result["files"] == 1
    assert [path.name for path in (tmp_path / "out").iterdir()] == ["a.json"]


def evidence_with(decision, codes, ecosystem="MAVEN"):
    base = json.loads((evidence.CONTRACT_DIR / "fixtures" / "valid" / "evidence_reject.json").read_text())
    base["request"]["change"] = {"ecosystem": ecosystem, "package": "org.example:lib" if ecosystem == "MAVEN" else "jinja2", "from_version": "1.0.0",
                                 "to_version": "2.0.0"}
    base["verdict"] = {"decision": decision, "findings": [{"code": code, "severity": "BLOCKING", "subject": "s", "message": "m", "evidence_ids": [],
                                                           "provenance": "p"} for code in codes]}
    return base


@pytest.mark.parametrize("decision,codes,expected", [
    ("ACCEPT", ["no-blocking-evidence"], "OBSERVED_PASS"),
    ("REJECT", ["test-regression"], "OBSERVED_REGRESSION"),
    ("REJECT", ["upgrade-compile-failed"], "OBSERVED_REGRESSION"),
    ("REJECT", ["new-vulnerability"], "INCONCLUSIVE"),
    ("REJECT", ["baseline-build-failed"], "INCONCLUSIVE"),
    ("INSUFFICIENT_EVIDENCE", ["tests-did-not-run"], "INCONCLUSIVE"),
])
def test_outcome_comes_from_the_verdict_findings(decision, codes, expected):
    assert evidence.outcome_of(evidence_with(decision, codes)) == expected


def test_policy_only_rejection_is_not_a_build_label():
    assert evidence.outcome_of(evidence_with("REJECT", ["licence-policy-violation", "new-vulnerability"])) == "INCONCLUSIVE"


def test_evidence_import_validates_against_the_contract_and_keeps_maven_only(tmp_path):
    good = tmp_path / "run-a"
    good.mkdir()
    (good / "evidence.json").write_text(json.dumps(evidence_with("ACCEPT", ["no-blocking-evidence"])))
    python = tmp_path / "run-b"
    python.mkdir()
    (python / "evidence.json").write_text(json.dumps(evidence_with("ACCEPT", ["no-blocking-evidence"], ecosystem="PYTHON")))
    broken = tmp_path / "run-c"
    broken.mkdir()
    (broken / "evidence.json").write_text("{\"schema_version\": \"1.0.0\"}")
    rows, rejected = evidence.load_evidence_rows(tmp_path)
    assert [row.outcome for row in rows] == ["OBSERVED_PASS"]
    assert rows[0].label_source == "sentry_evidence" and rows[0].provenance["evidence_sha256"]
    assert len(rejected) == 2


def test_three_real_sentry_runs_import_with_the_expected_outcomes():
    from pathlib import Path
    folder = Path(__file__).parent / "fixtures" / "sentry_evidence"
    rows, rejected = evidence.load_evidence_rows(folder)
    assert not rejected
    assert {row.to_version: row.outcome for row in rows} == {"1.1.0": "OBSERVED_PASS", "1.2.0": "OBSERVED_REGRESSION", "2.0.0": "OBSERVED_REGRESSION"}
    assert all(row.provenance["locally_verified"] for row in rows)
