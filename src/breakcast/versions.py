import re
from dataclasses import dataclass

VERSION = re.compile(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:[.-]?(.*))?$")
PRERELEASE_WORDS = ("alpha", "beta", "rc", "snapshot", "milestone", "preview", "cr")


@dataclass(frozen=True)
class Version:
    major: int
    minor: int
    patch: int
    suffix: str
    parsed: bool


def parse(text):
    match = VERSION.match(text.strip())
    if not match:
        return Version(0, 0, 0, text, False)
    major, minor, patch, suffix = match.groups()
    return Version(int(major), int(minor or 0), int(patch or 0), (suffix or "").lower(), True)


def is_prerelease(version):
    lowered = version.suffix
    return any(word in lowered for word in PRERELEASE_WORDS) or bool(re.search(r"(^|[.-])m\d+", lowered))


def change_kind(before, after):
    if not (before.parsed and after.parsed):
        return "unparsed"
    if after.major != before.major:
        return "major"
    if after.minor != before.minor:
        return "minor"
    if after.patch != before.patch:
        return "patch"
    return "other"


def is_downgrade(before, after):
    return (after.major, after.minor, after.patch) < (before.major, before.minor, before.patch)
