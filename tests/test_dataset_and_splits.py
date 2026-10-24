import pytest

from breakcast import dataset, splits
from breakcast.dataset import InvalidRow, Row


def make_row(**changes):
    base = dict(row_id="r1", project="org/proj", package="g:a", from_version="1.0.0", to_version="2.0.0", outcome="OBSERVED_REGRESSION",
                label_source="fixture", features={"major_delta": 1}, missing_features=[])
    base.update(changes)
    return Row(**base)


def test_a_bump_record_can_never_be_a_pass():
    with pytest.raises(InvalidRow):
        make_row(outcome="OBSERVED_PASS", label_source="bump_metadata").validate()


def test_empty_features_must_be_listed_as_missing():
    with pytest.raises(InvalidRow):
        make_row(features={"major_delta": None}).validate()
    make_row(features={"major_delta": None}, missing_features=["major_delta"]).validate()


def test_bad_package_and_schema_version_are_refused():
    with pytest.raises(InvalidRow):
        make_row(package="nocolon").validate()
    with pytest.raises(InvalidRow):
        make_row(feature_schema_version="0.0.1").validate()


def test_roundtrip_and_line_error_report(tmp_path):
    path = tmp_path / "rows.jsonl"
    dataset.write_rows([make_row(), make_row(row_id="r2", package="g:b")], path)
    assert len(dataset.read_rows(path)) == 2
    path.write_text(path.read_text() + "not json\n")
    with pytest.raises(InvalidRow, match="line 3"):
        dataset.read_rows(path)


def test_duplicates_collapse_and_evidence_wins_over_metadata():
    metadata = make_row(row_id="a", label_source="bump_metadata")
    verified = make_row(row_id="b", label_source="sentry_evidence")
    kept = dataset.deduplicate([metadata, make_row(row_id="c", label_source="bump_metadata"), verified])
    assert len(kept) == 1 and kept[0].label_source == "sentry_evidence"


def test_forks_with_different_case_share_a_group():
    assert make_row(project="Org/Proj").group == make_row(project="org/proj").group


def test_every_project_lands_in_exactly_one_partition(fixture_data):
    rows, manifest = fixture_data
    parts = splits.split_rows(rows, manifest)
    seen = {name: {row.group for row in parts[name]} for name in parts}
    assert not seen["train"] & seen["dev"] and not seen["train"] & seen["test"] and not seen["dev"] & seen["test"]
    assert sum(len(items) for items in parts.values()) == len(rows)


def test_partitions_do_not_move_when_projects_are_added(fixture_data):
    rows, manifest = fixture_data
    extra = make_row(row_id="new", project="brand/new")
    extended = splits.extend_manifest(manifest, [extra])
    for group, name in manifest["assignment"].items():
        assert extended["assignment"][group] == name


def test_tampered_manifest_is_refused(tmp_path, fixture_data):
    _, manifest = fixture_data
    path = tmp_path / "splits.json"
    splits.save_manifest(manifest, path)
    path.write_text(path.read_text().replace('"train"', '"test"', 1))
    with pytest.raises(splits.SplitMismatch):
        splits.load_manifest(path)


def test_project_missing_from_the_manifest_is_an_error(fixture_data):
    _, manifest = fixture_data
    with pytest.raises(splits.SplitMismatch):
        splits.split_rows([make_row(project="unknown/proj")], manifest)


def test_inventory_counts(fixture_data):
    rows, _ = fixture_data
    counts = dataset.inventory(rows)
    assert counts["rows"] == len(rows) and counts["by_source"] == {"fixture": len(rows)}
