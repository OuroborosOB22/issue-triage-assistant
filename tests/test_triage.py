import json

import pytest
from fastapi.testclient import TestClient

from triage import artifacts, data, metrics, models, train
from triage.service import create_app

WORDS = {"bug": "crash error exception fails broken stacktrace", "feature": "add support request proposal enhancement option",
         "question": "how can i what is why does help usage"}


def make_issues(per_label=30, repos=("a/one", "b/two")):
    issues = []
    for index in range(per_label):
        for label, words in WORDS.items():
            title = "%s %d %s" % (words.split()[index % 6], index, label)
            issues.append(data.Issue(repos[index % len(repos)], label, title, "%s details number %d" % (words, index)))
    return issues


@pytest.fixture
def release(tmp_path):
    issues = make_issues()
    fit_issues, dev_issues = data.dev_split(issues)
    model = models.fit(models.tfidf_logreg(3.0), fit_issues)
    scores = models.predict_scores(model, [issue.text for issue in dev_issues])
    threshold = metrics.review_threshold([issue.label for issue in dev_issues], scores)
    directory = tmp_path / "release"
    artifacts.save_sklearn(directory, model, "tfidf_logreg", "test-1", {"macro_f1": 1.0}, {"source": "made up"}, threshold)
    return directory


def test_clean_train_drops_rows_that_are_in_the_test_set_or_repeated():
    a, b, c = (data.Issue("r", "bug", t, "x") for t in ("one", "two", "three"))
    kept, dropped = data.clean_train([a, a, b, c], [b])
    assert kept == [a, c]
    assert dropped == {"duplicate": 1, "in_test": 1}


def test_dev_split_is_stratified_seeded_and_disjoint():
    issues = make_issues(per_label=20)
    fit_a, dev_a = data.dev_split(issues)
    fit_b, dev_b = data.dev_split(issues)
    assert dev_a == dev_b
    assert not set(fit_a) & set(dev_a)
    assert len(fit_a) + len(dev_a) == len(issues)
    assert {issue.label for issue in dev_a} == set(data.LABELS)


def test_read_refuses_unknown_labels(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("repo,created_at,label,title,body\nr,2020,enhancement,t,b\n")
    with pytest.raises(data.BadData):
        data.read(path)


def test_metrics_report_macro_f1_and_confusion_matrix():
    truth = ["bug", "bug", "feature", "question"]
    scores = [{"bug": .9, "feature": .05, "question": .05}, {"bug": .2, "feature": .7, "question": .1},
              {"bug": .1, "feature": .8, "question": .1}, {"bug": .1, "feature": .1, "question": .8}]
    report = metrics.evaluate(truth, scores, ["r"] * 4)
    assert report["confusion_matrix"]["matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert 0 < report["macro_f1"] < 1
    assert report["per_repo_macro_f1"]["r"] == report["cross_repo_mean_f1"]


def test_review_threshold_defers_low_confidence_cases():
    truth = ["bug", "bug", "bug", "bug"]
    scores = [{"bug": .95, "feature": .03, "question": .02}, {"bug": .9, "feature": .05, "question": .05},
              {"bug": .4, "feature": .5, "question": .1}, {"bug": .35, "feature": .5, "question": .15}]
    threshold = metrics.review_threshold(truth, scores, target_accuracy=0.9)
    assert threshold == 0.9
    assert metrics.coverage(truth, scores, threshold)["accepted"] == 0.5


def test_release_loads_and_rejects_changed_files(release):
    predictor, manifest = artifacts.load(release)
    assert manifest["model_version"] == "test-1" and predictor.scores(["crash error"])[0]
    (release / artifacts.SKLEARN_FILE).write_bytes(b"changed")
    with pytest.raises(artifacts.UntrustedArtifact):
        artifacts.load(release)


def test_final_test_runs_once(release, tmp_path):
    predictor, manifest = artifacts.load(release)
    test_issues = make_issues(per_label=5)
    report = train.final_test(predictor, manifest, "digest", test_issues, tmp_path / "reports")
    assert report["manifest_sha256"] == "digest" and report["rows"] == len(test_issues)
    with pytest.raises(FileExistsError):
        train.final_test(predictor, manifest, "digest", test_issues, tmp_path / "reports")


@pytest.fixture
def client(release, tmp_path):
    return TestClient(create_app(release, tmp_path / "none.json"))


def test_single_prediction_returns_label_score_and_review(client):
    answer = client.post("/v1/classify", json={"title": "app crash error", "body": "exception stacktrace"}).json()
    assert answer["label"] == "bug" and 0 < answer["score"] <= 1 and answer["review"] in ("ok", "needs_review", None)
    assert answer["model_version"] == "test-1"


@pytest.mark.parametrize("payload", [{}, {"title": ""}, {"title": "x" * 301}, {"title": "ok", "body": 5}, [1]])
def test_single_prediction_validates_input(client, payload):
    assert client.post("/v1/classify", json=payload).status_code == 422


def test_model_info_ready_and_report_states(client):
    assert client.get("/ready").status_code == 200
    assert client.get("/v1/model-info").json()["calibrated"] is False
    assert client.get("/v1/report").status_code == 404


def test_service_without_a_model_says_unavailable(tmp_path):
    app = TestClient(create_app(tmp_path / "missing", tmp_path / "none.json"))
    assert app.get("/ready").status_code == 503
    assert app.post("/v1/classify", json={"title": "t"}).status_code == 503


def test_batch_returns_per_row_results_and_errors(client):
    body = "title,body\napp crash error,exception\n,empty title\nhow can i help,what is usage\n"
    answer = client.post("/v1/classify-batch", content=body, headers={"content-type": "text/csv"}).json()
    assert answer["count"] == 3
    assert answer["results"][0]["label"] == "bug"
    assert "error" in answer["results"][1]
    assert answer["results"][2]["label"] == "question"


def test_batch_needs_a_title_column_and_limits_rows(client):
    assert client.post("/v1/classify-batch", content="name\nx\n").status_code == 422
    many = "title\n" + "x\n" * 501
    assert client.post("/v1/classify-batch", content=many).status_code == 413
    assert client.post("/v1/classify-batch", content=b"\xff\xfe\x00bad").status_code == 422


def test_csv_export_neutralises_spreadsheet_formulas(client):
    body = 'title,body\n"=HYPERLINK(""http://x"")",crash error\n'
    text = client.post("/v1/classify-batch?format=csv", content=body).text
    assert "'=HYPERLINK" in text and "\n=HYPERLINK" not in text


def test_responses_carry_security_headers(client):
    headers = client.get("/health").headers
    assert "default-src 'none'" in headers["content-security-policy"] and headers["x-content-type-options"] == "nosniff"


def test_the_page_and_its_assets_are_served_with_the_strict_policy(client):
    page = client.get("/")
    assert page.status_code == 200 and "Is it a bug" in page.text
    assert client.get("/static/app.js").status_code == 200 and client.get("/static/style.css").status_code == 200
    assert client.get("/static/fonts/inter-latin-400-normal.woff2").status_code == 200
    assert "script-src 'self'" in page.headers["content-security-policy"] and "unsafe-inline" not in page.headers["content-security-policy"]


def test_the_page_has_no_inline_script_or_remote_resources(client):
    html = client.get("/").text
    assert "<script>" not in html and "onclick=" not in html and "https://" not in html and "http://" not in html


def test_a_saved_test_report_is_only_shown_for_the_release_it_scored(release, tmp_path):
    report = tmp_path / "final_test.json"
    report.write_text(json.dumps({"model_version": "some-other-release", "macro_f1": 0.99}))
    other = TestClient(create_app(release, report))
    assert other.get("/v1/report").status_code == 404
    report.write_text(json.dumps({"model_version": "test-1", "macro_f1": 0.5}))
    assert TestClient(create_app(release, report)).get("/v1/report").json()["macro_f1"] == 0.5


def test_the_development_comparison_is_served_when_it_was_saved(release, tmp_path):
    (tmp_path / "dev_comparison.json").write_text(json.dumps({"chosen": "x", "models": {}}))
    assert TestClient(create_app(release, tmp_path / "final_test.json")).get("/v1/comparison").json()["chosen"] == "x"
    assert TestClient(create_app(release, tmp_path / "nowhere" / "final_test.json")).get("/v1/comparison").status_code == 404


def test_the_logo_is_served_and_used_as_the_tab_icon(client):
    assert 'rel="icon"' in client.get("/").text
    assert client.get("/static/logo.png").headers["content-type"] == "image/png"
