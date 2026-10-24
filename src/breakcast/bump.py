import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from .dataset import Row
from .features import extract

REPOSITORY = "https://github.com/chains-project/bump.git"
BENCHMARK = "data/benchmark"
TIMEOUT = 300


def run_git(arguments, cwd=None):
    result = subprocess.run(["git", *arguments], cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError("git %s failed: %s" % (arguments[0], result.stderr.strip()[:300]))


def fetch_benchmark(destination, repository=REPOSITORY):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as work:
        clone = Path(work) / "bump"
        run_git(["clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", repository, str(clone)])
        run_git(["sparse-checkout", "set", BENCHMARK], cwd=clone)
        files = sorted((clone / BENCHMARK).glob("*.json"))
        for path in files:
            shutil.copyfile(path, destination / path.name)
    return {"files": len(files), "into": str(destination)}


def project_name(record):
    return "%s/%s" % (record["projectOrganisation"], record["project"])


def row_from_record(record):
    update = record["updatedDependency"]
    package = "%s:%s" % (update["dependencyGroupID"], update["dependencyArtifactID"])
    features, missing = extract(update["previousVersion"], update["newVersion"], update.get("dependencyScope"), update.get("dependencySection"),
                                update.get("updatedFileType"))
    return Row(
        row_id="bump-" + record["breakingCommit"][:12],
        project=project_name(record),
        package=package,
        from_version=update["previousVersion"],
        to_version=update["newVersion"],
        outcome="OBSERVED_REGRESSION",
        label_source="bump_metadata",
        features=features,
        missing_features=missing,
        provenance={"source": "chains-project/bump", "pull_request": record.get("url"), "breaking_commit": record["breakingCommit"],
                    "reproduction_image": record.get("breakingUpdateReproductionCommand"), "java_version": record.get("javaVersionUsedForReproduction"),
                    "locally_verified": False},
    )


def load_rows(directory):
    rows, skipped = [], []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            record = json.loads(path.read_text())
            if "updatedDependency" not in record:
                raise KeyError("updatedDependency")
            rows.append(row_from_record(record))
        except (KeyError, ValueError, TypeError) as problem:
            skipped.append({"file": path.name, "reason": str(problem)})
    return rows, skipped
