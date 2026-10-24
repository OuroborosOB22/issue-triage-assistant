import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy
import pandas
import sklearn

from . import FEATURE_SCHEMA_VERSION, TASK

MODEL_FILE = "model.joblib"
MANIFEST_FILE = "manifest.json"
SUPPORTED_SCOPE = "Maven version-only upgrades, metadata features only (schema %s)" % FEATURE_SCHEMA_VERSION


class UntrustedArtifact(RuntimeError):
    pass


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(directory, model, model_name, model_version, metrics, data_digest, split_digest, is_fixture, extra=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / MODEL_FILE)
    manifest = {
        "task": TASK,
        "model_name": model_name,
        "model_version": model_version,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "supported_scope": ("FIXTURE MODEL, built from a tiny labelled test set, not a research result. " if is_fixture else "") + SUPPORTED_SCOPE,
        "is_fixture": is_fixture,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_digest": data_digest,
        "split_digest": split_digest,
        "metrics": metrics,
        "calibration_assessed": False,
        "packages": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": numpy.__version__, "pandas": pandas.__version__,
                     "joblib": joblib.__version__},
        "model_sha256": sha256_of(directory / MODEL_FILE),
    }
    manifest.update(extra or {})
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def load(directory, pinned_sha256=None):
    directory = Path(directory)
    manifest_path, model_path = directory / MANIFEST_FILE, directory / MODEL_FILE
    if not manifest_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("no model release in %s" % directory)
    manifest = json.loads(manifest_path.read_text())
    digest = sha256_of(model_path)
    if digest != manifest.get("model_sha256"):
        raise UntrustedArtifact("the model file does not match its manifest")
    if pinned_sha256 and digest != pinned_sha256:
        raise UntrustedArtifact("the model file does not match the pinned digest")
    if manifest.get("task") != TASK or manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise UntrustedArtifact("the model was built for a different task or feature schema")
    return joblib.load(model_path), manifest
