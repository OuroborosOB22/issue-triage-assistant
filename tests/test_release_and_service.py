import json

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from breakcast import artifacts
from breakcast.evidence import CONTRACT_DIR, registry
from breakcast.service import create_app


def response_check():
    schema = json.loads((CONTRACT_DIR / "prediction_response.schema.json").read_text())
    return Draft202012Validator(schema, registry=registry())


def request_body(**changes):
    body = json.loads((CONTRACT_DIR / "fixtures" / "valid" / "prediction_request.json").read_text())
    body["feature_schema_version"] = "1.0.0"
    body["features"] = {"major_delta": 1, "minor_delta": 0, "patch_delta": 0, "change_kind": "major", "scope": "compile"}
    body["missing_features"] = ["from_prerelease"]
    body.update(changes)
    return body


@pytest.fixture
def client(fixture_release):
    return TestClient(create_app(fixture_release))


@pytest.fixture
def empty_client():
    return TestClient(create_app(None))


def test_release_roundtrip_and_manifest(fixture_release):
    model, manifest = artifacts.load(fixture_release)
    assert manifest["is_fixture"] is True and "FIXTURE" in manifest["supported_scope"]
    assert manifest["calibration_assessed"] is False
    assert model is not None


def test_changed_model_file_is_refused(fixture_release, tmp_path):
    copy = tmp_path / "copy"
    copy.mkdir()
    for name in (artifacts.MODEL_FILE, artifacts.MANIFEST_FILE):
        (copy / name).write_bytes((fixture_release / name).read_bytes())
    (copy / artifacts.MODEL_FILE).write_bytes(b"swapped")
    with pytest.raises(artifacts.UntrustedArtifact):
        artifacts.load(copy)


def test_pinned_digest_must_match(fixture_release):
    with pytest.raises(artifacts.UntrustedArtifact):
        artifacts.load(fixture_release, pinned_sha256="0" * 64)


def test_wrong_feature_schema_in_manifest_is_refused(fixture_release, tmp_path):
    copy = tmp_path / "copy"
    copy.mkdir()
    manifest = json.loads((fixture_release / artifacts.MANIFEST_FILE).read_text())
    manifest["feature_schema_version"] = "9.9.9"
    (copy / artifacts.MANIFEST_FILE).write_text(json.dumps(manifest))
    (copy / artifacts.MODEL_FILE).write_bytes((fixture_release / artifacts.MODEL_FILE).read_bytes())
    with pytest.raises(artifacts.UntrustedArtifact):
        artifacts.load(copy)


def test_health_and_ready_are_different(empty_client, client):
    assert empty_client.get("/health").status_code == 200
    assert empty_client.get("/ready").status_code == 503
    assert client.get("/ready").status_code == 200


def test_model_info_says_it_is_a_fixture(client):
    info = client.get("/v1/model-info").json()
    assert info["is_fixture"] is True and info["model_version"] == "fixture-test"


def test_predict_returns_a_valid_contract_response(client):
    reply = client.post("/v1/predict", json=request_body())
    assert reply.status_code == 200
    body = reply.json()
    assert not list(response_check().iter_errors(body))
    assert body["status"] == "OK" and 0 <= body["risk_score"] <= 1
    assert body["calibrated_probability"] is None
    assert "FIXTURE" in body["supported_scope"]


def test_a_bigger_jump_scores_at_least_as_risky_as_a_patch(client):
    major = client.post("/v1/predict", json=request_body()).json()["risk_score"]
    patch = client.post("/v1/predict", json=request_body(features={"major_delta": 0, "minor_delta": 0, "patch_delta": 1, "change_kind": "patch",
                                                                   "scope": "compile"})).json()["risk_score"]
    assert major >= patch


def test_without_a_model_the_service_says_so_and_invents_no_score(empty_client):
    reply = empty_client.post("/v1/predict", json=request_body())
    body = reply.json()
    assert reply.status_code == 503 and body["status"] == "MODEL_UNAVAILABLE" and body["risk_score"] is None
    assert not list(response_check().iter_errors(body))


def test_unsupported_ecosystem(client):
    reply = client.post("/v1/predict", json=request_body(ecosystem="PYTHON"))
    assert reply.status_code == 422 and reply.json()["status"] == "UNSUPPORTED_ECOSYSTEM"
    assert not list(response_check().iter_errors(reply.json()))


@pytest.mark.parametrize("changes", [
    {"feature_schema_version": "2.0.0"},
    {"features": {"surprise": 1}},
    {"features": {"scope": ["list"]}},
    {"package": "no colon"},
    {"extra": 1},
])
def test_invalid_requests_are_refused_clearly(client, changes):
    reply = client.post("/v1/predict", json=request_body(**changes))
    assert reply.status_code == 422 and reply.json()["status"] == "INVALID_FEATURES" and reply.json()["risk_score"] is None
    assert not list(response_check().iter_errors(reply.json()))


def test_not_json_and_not_an_object(client):
    assert client.post("/v1/predict", content=b"nope").status_code == 422
    assert client.post("/v1/predict", json=[1, 2]).status_code == 422


def test_oversized_body_is_refused(client):
    assert client.post("/v1/predict", content=b"{" + b" " * 30000 + b"}").status_code == 413


def test_a_chunked_oversized_body_is_cut_off(client):
    def chunks():
        for _ in range(40):
            yield b" " * 1000
    assert client.post("/v1/predict", content=chunks()).status_code == 413


def test_no_docs_or_openapi_pages(client):
    assert client.get("/docs").status_code == 404 and client.get("/openapi.json").status_code == 404


def test_tampered_release_leaves_the_service_not_ready(fixture_release, tmp_path):
    copy = tmp_path / "copy"
    copy.mkdir()
    for name in (artifacts.MODEL_FILE, artifacts.MANIFEST_FILE):
        (copy / name).write_bytes((fixture_release / name).read_bytes())
    (copy / artifacts.MODEL_FILE).write_bytes(b"swapped")
    tampered = TestClient(create_app(copy))
    assert tampered.get("/ready").status_code == 503
    assert tampered.post("/v1/predict", json=request_body()).status_code == 503
