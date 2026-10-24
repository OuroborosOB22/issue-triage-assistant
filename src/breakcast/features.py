from . import FEATURE_SCHEMA_VERSION
from .versions import change_kind, is_downgrade, is_prerelease, parse

NUMERIC = ["major_delta", "minor_delta", "patch_delta", "from_prerelease", "to_prerelease", "is_downgrade", "version_parsed"]
CATEGORICAL = ["change_kind", "scope", "section", "file_type"]
ALL = NUMERIC + CATEGORICAL
SCOPES = ["compile", "test", "runtime", "provided", "other"]
SECTIONS = ["dependencies", "dependencyManagement", "plugins", "parent", "properties", "other"]
FILE_TYPES = ["JAR", "POM", "other"]
VOCABULARY = {"change_kind": ["major", "minor", "patch", "other", "unparsed"], "scope": SCOPES, "section": SECTIONS, "file_type": FILE_TYPES}


def known_or_other(value, allowed):
    return value if value in allowed else "other"


def extract(from_version, to_version, scope=None, section=None, file_type=None):
    before = parse(from_version)
    after = parse(to_version)
    parsed = before.parsed and after.parsed
    features = {
        "major_delta": after.major - before.major if parsed else None,
        "minor_delta": after.minor - before.minor if parsed else None,
        "patch_delta": after.patch - before.patch if parsed else None,
        "from_prerelease": int(is_prerelease(before)),
        "to_prerelease": int(is_prerelease(after)),
        "is_downgrade": int(is_downgrade(before, after)) if parsed else None,
        "version_parsed": int(parsed),
        "change_kind": change_kind(before, after),
        "scope": known_or_other(scope, SCOPES) if scope else None,
        "section": known_or_other(section, SECTIONS) if section else None,
        "file_type": known_or_other(file_type, FILE_TYPES) if file_type else None,
    }
    missing = [name for name, value in features.items() if value is None]
    return features, missing


def schema_description():
    return {"feature_schema_version": FEATURE_SCHEMA_VERSION, "numeric": NUMERIC, "categorical": CATEGORICAL, "vocabulary": VOCABULARY}
