import json
from pathlib import Path

import pytest

from breakcast import artifacts, dataset, fixture, splits, train

BUMP_RECORD = {
    "url": "https://github.com/example/project/pull/1",
    "project": "project",
    "projectOrganisation": "example",
    "breakingCommit": "0123456789abcdef0123456789abcdef01234567",
    "updatedDependency": {
        "dependencyGroupID": "org.apache.mina",
        "dependencyArtifactID": "mina-core",
        "previousVersion": "2.1.5",
        "newVersion": "2.2.1",
        "dependencyScope": "compile",
        "versionUpdateType": "minor",
        "updatedFileType": "JAR",
        "dependencySection": "dependencies",
    },
    "breakingUpdateReproductionCommand": "docker run example/image",
    "javaVersionUsedForReproduction": "11",
    "failureCategory": "COMPILATION_FAILURE",
}


@pytest.fixture
def bump_record():
    return json.loads(json.dumps(BUMP_RECORD))


@pytest.fixture(scope="session")
def fixture_release(tmp_path_factory):
    rows = fixture.fixture_rows()
    manifest = splits.build_manifest(rows)
    models, report = train.train_all(rows, manifest, min_per_class=20, min_projects=5)
    chosen = train.choose(report)
    directory = tmp_path_factory.mktemp("release")
    artifacts.save(directory, models[chosen], chosen, "fixture-test", report[chosen], "digest", manifest["digest"], True)
    return directory


@pytest.fixture(scope="session")
def fixture_data():
    rows = fixture.fixture_rows()
    return rows, splits.build_manifest(rows)
