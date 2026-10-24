import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import FEATURE_SCHEMA_VERSION
from .artifacts import UntrustedArtifact, load
from .evidence import validator
from .features import ALL
from .models import contributions, failure_scores, to_frame

MAX_BODY = 20000


def response_body(status, manifest, score=None, explanations=None, scope=None, inference_ms=0.0):
    return {
        "schema_version": "1.0.0",
        "status": status,
        "model_version": manifest["model_version"] if manifest and status == "OK" else None,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "risk_score": score,
        "calibrated_probability": None,
        "supported_scope": scope or (manifest["supported_scope"] if manifest else "maven only"),
        "explanations": explanations or [],
        "timing_ms": {"inference": round(inference_ms, 3)},
    }


def create_app(model_dir=None, pinned_sha256=None):
    app = FastAPI(title="BreakCast", docs_url=None, redoc_url=None, openapi_url=None)
    state = {"model": None, "manifest": None, "problem": None}
    directory = model_dir or os.environ.get("BREAKCAST_MODEL_DIR")
    pin = pinned_sha256 or os.environ.get("BREAKCAST_MODEL_SHA256")
    if directory:
        try:
            state["model"], state["manifest"] = load(directory, pin)
        except (FileNotFoundError, UntrustedArtifact) as problem:
            state["problem"] = str(problem)
    check = validator("prediction_request", Path(__file__).resolve().parents[2] / "contracts" / "v1")

    @app.get("/health")
    def health():
        return {"status": "alive"}

    @app.get("/ready")
    def ready():
        if state["model"] is None:
            return JSONResponse({"status": "not ready", "reason": state["problem"] or "no model release is configured"}, status_code=503)
        return {"status": "ready"}

    @app.get("/v1/model-info")
    def model_info():
        if state["manifest"] is None:
            return JSONResponse({"status": "MODEL_UNAVAILABLE", "reason": state["problem"] or "no model release is configured"}, status_code=503)
        keep = ("task", "model_name", "model_version", "feature_schema_version", "supported_scope", "is_fixture", "created_at", "data_digest",
                "split_digest", "calibration_assessed", "metrics")
        return {key: state["manifest"].get(key) for key in keep}

    @app.post("/v1/predict")
    async def predict(request: Request):
        raw = await request.body()
        if len(raw) > MAX_BODY:
            return JSONResponse(response_body("INVALID_FEATURES", state["manifest"], scope="request too large"), status_code=413)
        try:
            data = json.loads(raw)
        except ValueError:
            return JSONResponse(response_body("INVALID_FEATURES", state["manifest"], scope="body is not JSON"), status_code=422)
        if isinstance(data, dict) and data.get("ecosystem") not in (None, "MAVEN"):
            return JSONResponse(response_body("UNSUPPORTED_ECOSYSTEM", state["manifest"], scope="maven only"), status_code=422)
        problems = [error.message for error in check.iter_errors(data)] if isinstance(data, dict) else ["body must be an object"]
        if problems:
            return JSONResponse(response_body("INVALID_FEATURES", state["manifest"], scope=problems[0][:180]), status_code=422)
        if state["model"] is None:
            return JSONResponse(response_body("MODEL_UNAVAILABLE", None, scope=state["problem"] or "no model release is configured"), status_code=503)
        if data["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
            return JSONResponse(response_body("INVALID_FEATURES", state["manifest"], scope="feature schema %s is not served" % data["feature_schema_version"]),
                                status_code=422)
        unknown = sorted(set(data["features"]) - set(ALL))
        if unknown:
            return JSONResponse(response_body("INVALID_FEATURES", state["manifest"], scope="unknown features: %s" % ", ".join(unknown[:5])), status_code=422)
        started = time.perf_counter()
        features = {name: data["features"].get(name) for name in ALL}
        score = float(failure_scores(state["model"], to_frame([features]))[0])
        explanations = [{"feature": name, "value": features[name]} for name in ALL if features[name] is not None]
        weights = contributions(state["model"], features)
        for item in explanations:
            if item["feature"] in weights:
                item["contribution"] = round(weights[item["feature"]], 6)
        missing = sorted(set(data["missing_features"]) | {name for name in ALL if features[name] is None})
        scope = state["manifest"]["supported_scope"] + (" Missing: %s." % ", ".join(missing) if missing else "")
        return response_body("OK", state["manifest"], score, explanations, scope[:200], (time.perf_counter() - started) * 1000)

    return app
