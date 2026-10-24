import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib

from . import SCHEMA_VERSION, TASK, models

MANIFEST = "manifest.json"
SKLEARN_FILE = "model.joblib"
SETFIT_DIR = "setfit"


class UntrustedArtifact(RuntimeError):
    pass


def file_hashes(directory):
    directory = Path(directory)
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*")) if path.is_file() and path.name != MANIFEST}


def write_manifest(directory, kind, name, version, metrics, dataset, review_threshold, extra=None):
    manifest = {
        "task": TASK, "schema_version": SCHEMA_VERSION, "kind": kind, "model_name": name, "model_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "dataset": dataset, "dev_metrics": metrics,
        "review_threshold": review_threshold, "calibrated": False, "python": platform.python_version(), "files": file_hashes(directory),
    }
    manifest.update(extra or {})
    (Path(directory) / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def save_sklearn(directory, model, name, version, metrics, dataset, review_threshold, extra=None):
    Path(directory).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(directory) / SKLEARN_FILE)
    return write_manifest(directory, "sklearn", name, version, metrics, dataset, review_threshold, extra)


def save_setfit(directory, model, name, version, metrics, dataset, review_threshold, extra=None):
    Path(directory).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(Path(directory) / SETFIT_DIR))
    return write_manifest(directory, "setfit", name, version, metrics, dataset, review_threshold, extra)


def verify(directory, pinned_manifest_sha256=None):
    directory = Path(directory)
    path = directory / MANIFEST
    if not path.is_file():
        raise FileNotFoundError("no model release in %s" % directory)
    if pinned_manifest_sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != pinned_manifest_sha256:
        raise UntrustedArtifact("the manifest does not match the pinned digest")
    manifest = json.loads(path.read_text())
    if manifest.get("task") != TASK:
        raise UntrustedArtifact("the release was built for a different task")
    if file_hashes(directory) != manifest.get("files"):
        raise UntrustedArtifact("the model files do not match their manifest")
    return manifest


def load(directory, pinned_manifest_sha256=None):
    manifest = verify(directory, pinned_manifest_sha256)
    directory = Path(directory)
    if manifest["kind"] == "sklearn":
        model = joblib.load(directory / SKLEARN_FILE)
        return Predictor(lambda texts: models.predict_scores(model, texts)), manifest
    from setfit import SetFitModel
    from . import setfit_model
    model = SetFitModel.from_pretrained(str(directory / SETFIT_DIR))
    return Predictor(lambda texts: setfit_model.predict_scores(model, texts)), manifest


class Predictor:
    def __init__(self, scorer):
        self.scorer = scorer

    def scores(self, texts):
        return self.scorer(texts)
