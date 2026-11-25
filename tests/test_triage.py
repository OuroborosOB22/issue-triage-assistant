import json
from pathlib import Path

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
    page = client.get("/app")
    assert page.status_code == 200 and "Is it a bug" in page.text
    assert client.get("/static/app.js").status_code == 200 and client.get("/static/style.css").status_code == 200
    assert client.get("/static/fonts/inter-latin-400-normal.woff2").status_code == 200
    assert "script-src 'self'" in page.headers["content-security-policy"] and "unsafe-inline" not in page.headers["content-security-policy"]


def test_the_page_has_no_inline_script_or_remote_resources(client):
    html = client.get("/app").text
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
    assert 'rel="icon"' in client.get("/app").text
    assert client.get("/static/logo.png").headers["content-type"] == "image/png"


def test_the_landing_page_is_served_and_leads_to_the_app(client):
    page = client.get("/")
    assert page.status_code == 200 and 'href="/app"' in page.text and 'href="#lifecycle"' in page.text
    assert "nc3" not in page.text and page.text.count('class="figbox nm') == 2 and "Gurman Singh and Madhur Tuteja" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert "<script>" not in page.text and "style=" not in page.text
    assert "https://" not in page.text and "http://" not in page.text
    for name in ("landing.css", "landing.js", "scene.png", "logo-64.png", "logo-128.png"):
        assert client.get("/static/" + name).status_code == 200


def test_the_landing_numbers_match_the_saved_report():
    report = json.loads((Path(__file__).resolve().parents[1] / "reports" / "final_test.json").read_text())
    page = (Path(__file__).resolve().parents[1] / "src" / "triage" / "web" / "landing.html").read_text()
    assert f"{report['macro_f1']:.3f}" in page and f"{report['accuracy']:.3f}" in page


def test_review_threshold_treats_tied_confidences_as_one_group():
    truth = ["bug"] * 10
    scores = [{"bug": 0.9, "feature": 0.05, "question": 0.05}] * 10
    wrong = {"bug": 0.05, "feature": 0.9, "question": 0.05}
    scores = scores[:8] + [wrong, wrong]
    threshold = metrics.review_threshold(truth, scores, target_accuracy=0.9)
    assert threshold is None
    assert metrics.coverage(truth, scores, threshold)["accepted"] == 0.0


def test_review_threshold_keeps_a_tied_group_that_is_good_enough():
    truth = ["bug"] * 10
    right = {"bug": 0.9, "feature": 0.05, "question": 0.05}
    wrong = {"bug": 0.05, "feature": 0.9, "question": 0.05}
    scores = [right] * 9 + [wrong]
    threshold = metrics.review_threshold(truth, scores, target_accuracy=0.9)
    assert threshold == 0.9
    assert metrics.coverage(truth, scores, threshold)["accuracy_on_accepted"] == 0.9


def test_review_threshold_prefers_the_lowest_confidence_that_still_meets_the_target():
    truth = ["bug"] * 4
    scores = [{"bug": 0.99, "feature": 0.0, "question": 0.01}, {"bug": 0.95, "feature": 0.0, "question": 0.05},
              {"bug": 0.9, "feature": 0.05, "question": 0.05}, {"bug": 0.2, "feature": 0.5, "question": 0.3}]
    assert metrics.review_threshold(truth, scores, target_accuracy=0.9) == 0.9


def test_missing_setfit_runtime_leaves_service_unavailable_not_crashed(tmp_path, monkeypatch):
    import sys
    directory = tmp_path / "setfit-release"
    (directory / artifacts.SETFIT_DIR).mkdir(parents=True)
    (directory / artifacts.SETFIT_DIR / "config.json").write_text("{}")
    artifacts.write_manifest(directory, "setfit", "setfit_test", "setfit-1", {"macro_f1": 0.8}, {"source": "test"}, 0.9)
    monkeypatch.setitem(sys.modules, "setfit", None)
    with pytest.raises(artifacts.UntrustedArtifact, match="setfit"):
        artifacts.load(directory)
    app = TestClient(create_app(directory, tmp_path / "none.json"))
    assert app.get("/ready").status_code == 503
    assert app.post("/v1/classify", json={"title": "crash bug"}).status_code == 503


def test_frontend_examples_are_valid_issue_payloads():
    import re
    from triage.service import parse_issue
    script = (Path(__file__).resolve().parents[1] / "src" / "triage" / "web" / "static" / "app.js").read_text()
    match = re.search(r"var EXAMPLES = (\[.*?\]);", script, re.DOTALL)
    assert match is not None
    items = re.findall(r'title:\s*"((?:[^"\\]|\\.)*)"\s*,\s*body:\s*"((?:[^"\\]|\\.)*)"', match.group(1))
    assert len(items) == 4
    for title, body in items:
        assert parse_issue({"title": title, "body": body.replace("\\n", "\n")})


def test_compare_setfit_passes_the_seed_to_training(monkeypatch):
    from triage import setfit_model
    seen = {}

    def fake_fit(issues, checkpoint, iterations, epochs, seed, device):
        seen.update(checkpoint=checkpoint, iterations=iterations, epochs=epochs, seed=seed, device=device)
        return "model"

    monkeypatch.setattr(setfit_model, "fit", fake_fit)
    monkeypatch.setattr(setfit_model, "predict_scores", lambda model, texts: [{"bug": 1.0, "feature": 0.0, "question": 0.0} for _ in texts])
    issues = make_issues(per_label=3)
    results = train.compare_setfit(issues, issues, ["org/small"], iterations=20, epochs=2, device="cuda", seed=23)
    assert seen == {"checkpoint": "org/small", "iterations": 20, "epochs": 2, "seed": 23, "device": "cuda"}
    assert results["setfit_small"]["dev"]["seed"] == 23


def test_unseen_study_uses_the_given_model_settings(monkeypatch, tmp_path):
    from triage import setfit_model, unseen
    seen = []
    monkeypatch.setattr(setfit_model, "fit", lambda issues, checkpoint, iterations, epochs, device: seen.append((checkpoint, iterations, epochs, device)) or "model")
    monkeypatch.setattr(setfit_model, "predict_scores", lambda model, texts: [{"bug": 1.0, "feature": 0.0, "question": 0.0} for _ in texts])
    issues = make_issues(per_label=30, repos=("a/one", "b/two"))
    monkeypatch.setattr(train, "prepare", lambda raw: {"fit": issues[:60], "dev": issues[60:], "test": make_issues(per_label=6, repos=("a/one", "b/two"))})
    out = unseen.run("raw", tmp_path, ["setfit"], "cuda", {"checkpoint": "org/big", "iterations": 20, "epochs": 2})
    assert seen and set(seen) == {("org/big", 20, 2, "cuda")}
    assert out["checkpoint"] == "org/big" and out["epochs"] == 2


def test_refit_on_all_counts_the_development_rows():
    from triage import cli
    prepared = {"fit": [1] * 10, "dev": [1] * 3, "test": [], "dropped": {}}
    assert cli.dataset_facts(prepared)["fit_rows"] == 10 and "refit_on_all" not in cli.dataset_facts(prepared)
    assert cli.dataset_facts(prepared, True)["fit_rows"] == 13 and cli.dataset_facts(prepared, True)["refit_on_all"] is True
