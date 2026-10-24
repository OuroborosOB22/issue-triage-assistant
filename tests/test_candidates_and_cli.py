import json

import pytest

from breakcast import candidates, cli


@pytest.mark.parametrize("title,expected", [
    ("Bump guava from 31.1-jre to 33.0.0-jre", ("guava", "31.1-jre", "33.0.0-jre")),
    ("chore(deps): bump org.slf4j:slf4j-api from 1.7.30 to 2.0.9", ("org.slf4j:slf4j-api", "1.7.30", "2.0.9")),
    ("Update jackson from 2.14.0 to 2.15.0", ("jackson", "2.14.0", "2.15.0")),
])
def test_bump_titles_are_parsed(title, expected):
    parsed = candidates.parse_title(title)
    assert (parsed["name"], parsed["before"], parsed["after"]) == expected


@pytest.mark.parametrize("title", ["Fix typo", "Bump versions and fix tests", "Bump guava from 1 to 2 and refactor"])
def test_other_titles_are_ignored(title):
    assert candidates.parse_title(title) is None


def test_only_single_pom_changes_are_candidates():
    assert candidates.single_dependency([{"filename": "pom.xml"}])
    assert candidates.single_dependency([{"filename": "module/pom.xml"}])
    assert not candidates.single_dependency([{"filename": "pom.xml"}, {"filename": "src/Main.java"}])


def test_mining_keeps_merged_single_dependency_pulls_only():
    pulls = [
        {"number": 1, "title": "Bump a from 1.0 to 1.1", "merged_at": "2024-01-01T00:00:00Z", "html_url": "u1", "base": {"sha": "b1"}, "head": {"sha": "h1"}},
        {"number": 2, "title": "Bump b from 1.0 to 1.1", "merged_at": None, "html_url": "u2", "base": {"sha": "b2"}, "head": {"sha": "h2"}},
        {"number": 3, "title": "Bump c from 1.0 to 1.1", "merged_at": "2024-01-02T00:00:00Z", "html_url": "u3", "base": {"sha": "b3"}, "head": {"sha": "h3"}},
    ]
    files = {1: [{"filename": "pom.xml"}], 3: [{"filename": "pom.xml"}, {"filename": "A.java"}]}

    def fake(path):
        if "/pulls?" in path:
            return pulls if "page=1" in path else []
        return files[int(path.split("/pulls/")[1].split("/")[0])]

    found = candidates.candidates_for("o/r", fake)
    assert [item["dependency"] for item in found] == ["a"]
    assert found[0]["base_commit"] == "b1"


def test_manifest_marks_candidates_as_not_verified(tmp_path):
    candidates.write_manifest([{"repo": "o/r", "dependency": "a", "from_version": "1", "to_version": "2"}], tmp_path / "m.jsonl")
    line = json.loads((tmp_path / "m.jsonl").read_text())
    assert line["status"] == "candidate-not-verified" and line["change"] == "version-only"


def test_cli_train_exits_with_an_error_when_data_is_too_thin(tmp_path, bump_record, capsys):
    folder = tmp_path / "bump"
    folder.mkdir()
    (folder / "a.json").write_text(json.dumps(bump_record))
    rows, splits = tmp_path / "rows.jsonl", tmp_path / "splits.json"
    assert cli.main(["build-dataset", "--bump", str(folder), "--out", str(rows), "--splits", str(splits)]) == 0
    assert cli.main(["train", "--rows", str(rows), "--splits", str(splits), "--out", str(tmp_path / "m")]) == 2
    assert "not enough verified two-class data" in capsys.readouterr().err
    assert not (tmp_path / "m").exists()


def test_cli_inventory_reports_the_blocker(tmp_path, bump_record, capsys):
    folder = tmp_path / "bump"
    folder.mkdir()
    (folder / "a.json").write_text(json.dumps(bump_record))
    rows, splits = tmp_path / "rows.jsonl", tmp_path / "splits.json"
    cli.main(["build-dataset", "--bump", str(folder), "--out", str(rows), "--splits", str(splits)])
    capsys.readouterr()
    cli.main(["inventory", "--rows", str(rows), "--splits", str(splits)])
    assert "needs at least" in capsys.readouterr().out
