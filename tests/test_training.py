import numpy as np
import pytest

from breakcast import metrics, models, splits, train
from breakcast.dataset import Row


def only_failures(count=40):
    return [Row(row_id="f%d" % n, project="org/p%d" % n, package="g:a", from_version="1.0.0", to_version="2.0.0", outcome="OBSERVED_REGRESSION",
                label_source="bump_metadata", features={"major_delta": 1}, missing_features=[]) for n in range(count)]


def test_training_refuses_when_there_are_no_passes():
    rows = only_failures()
    with pytest.raises(train.InsufficientData, match="0 passes"):
        train.train_all(rows, splits.build_manifest(rows))


def test_training_refuses_with_too_few_projects(fixture_data):
    rows, manifest = fixture_data
    with pytest.raises(train.InsufficientData, match="projects"):
        train.train_all(rows, manifest, min_per_class=1, min_projects=500)


def test_inconclusive_rows_never_reach_training(fixture_data):
    rows, manifest = fixture_data
    odd = Row(row_id="x", project=rows[0].project, package="g:a", from_version="1.0.0", to_version="2.0.0", outcome="INCONCLUSIVE", label_source="fixture",
              features=rows[0].features, missing_features=[])
    assert odd not in train.labelled(rows + [odd])


def test_all_four_models_train_and_are_scored_on_dev(fixture_data):
    rows, manifest = fixture_data
    trained, report = train.train_all(rows, manifest, min_per_class=20, min_projects=5)
    assert set(trained) == set(models.MODEL_NAMES)
    assert all("pr_auc" in report[name]["dev"] for name in trained)
    assert train.choose(report) in trained


def test_preprocessing_is_fitted_on_training_rows_only(fixture_data):
    rows, manifest = fixture_data
    trained, _ = train.train_all(rows, manifest, min_per_class=20, min_projects=5)
    parts = splits.split_rows(train.labelled(rows), manifest)
    frame = models.to_frame([row.features for row in parts["train"]])
    imputer = trained["logistic_regression"].named_steps["prepare"].named_transformers_["numeric"].named_steps["fill"]
    expected = np.nanmedian(frame[models.NUMERIC].to_numpy(dtype=float), axis=0)
    assert np.allclose(imputer.statistics_, expected)


def test_missing_features_do_not_break_scoring(fixture_data):
    rows, manifest = fixture_data
    trained, _ = train.train_all(rows, manifest, min_per_class=20, min_projects=5)
    frame = models.to_frame([{"change_kind": "major"}])
    for model in trained.values():
        assert 0.0 <= float(models.failure_scores(model, frame)[0]) <= 1.0


def test_the_rule_baseline_flags_only_major_jumps():
    frame = models.to_frame([{"change_kind": "major"}, {"change_kind": "minor"}, {"change_kind": "unparsed"}])
    assert list(models.MajorJumpRule().predict(frame)) == [1, 0, 0]


def test_final_test_reports_only_chosen_and_baselines(fixture_data):
    rows, manifest = fixture_data
    trained, report = train.train_all(rows, manifest, min_per_class=20, min_projects=5)
    chosen = train.choose(report)
    assert set(train.final_test(trained, rows, manifest, chosen)) <= {chosen, "majority", "major_jump_rule"}


def test_ranking_metrics_are_undefined_for_one_class():
    result = metrics.evaluate([1, 1, 1], [0.9, 0.8, 0.7], ["a", "b", "c"])
    assert result["pr_auc"] is None and result["roc_auc"] is None and result["undefined_note"]


def test_precision_is_undefined_when_nothing_is_flagged():
    result = metrics.evaluate([0, 1, 0, 1], [0.1, 0.2, 0.1, 0.3], ["a", "b", "c", "d"])
    assert result["precision"] is None and result["failure_recall"] == 0.0


def test_perfect_ranking_scores_one_and_confusion_matrix_adds_up():
    result = metrics.evaluate([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], ["a", "b", "c", "d"])
    assert result["pr_auc"] == 1.0 and result["roc_auc"] == 1.0
    assert sum(sum(line) for line in result["confusion_matrix"]["matrix"]) == 4


def test_interval_needs_enough_projects():
    assert metrics.group_interval([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], ["a", "b", "c", "d"]) is None
