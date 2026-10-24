import hashlib
import json
from pathlib import Path

SEED = "breakcast-split-1"
FRACTIONS = {"train": 0.6, "dev": 0.2, "test": 0.2}
PARTITIONS = ("train", "dev", "test")


class SplitMismatch(ValueError):
    pass


def partition_of(group, seed=SEED):
    digest = hashlib.sha256(("%s|%s" % (seed, group)).encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    if fraction < FRACTIONS["train"]:
        return "train"
    if fraction < FRACTIONS["train"] + FRACTIONS["dev"]:
        return "dev"
    return "test"


def build_manifest(rows, seed=SEED):
    groups = sorted({row.group for row in rows})
    assignment = {group: partition_of(group, seed) for group in groups}
    body = {"seed": seed, "fractions": FRACTIONS, "assignment": assignment}
    body["digest"] = digest_of(body)
    return body


def digest_of(body):
    plain = {key: value for key, value in body.items() if key != "digest"}
    return hashlib.sha256(json.dumps(plain, sort_keys=True).encode()).hexdigest()


def save_manifest(manifest, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("digest") != digest_of(manifest):
        raise SplitMismatch("the split manifest was changed after it was written")
    return manifest


def extend_manifest(manifest, rows):
    assignment = dict(manifest["assignment"])
    for row in rows:
        existing = assignment.get(row.group)
        expected = partition_of(row.group, manifest["seed"])
        if existing is not None and existing != expected:
            raise SplitMismatch("project %s moved from %s to %s" % (row.group, existing, expected))
        assignment[row.group] = expected
    body = {"seed": manifest["seed"], "fractions": manifest["fractions"], "assignment": assignment}
    body["digest"] = digest_of(body)
    return body


def split_rows(rows, manifest):
    parts = {name: [] for name in PARTITIONS}
    for row in rows:
        name = manifest["assignment"].get(row.group)
        if name is None:
            raise SplitMismatch("project %s is not in the split manifest" % row.group)
        parts[name].append(row)
    leaked = set.intersection(*[{row.group for row in parts[name]} for name in ("train", "dev")]) if all(parts[n] for n in ("train", "dev")) else set()
    if leaked:
        raise SplitMismatch("projects in two partitions: %s" % sorted(leaked))
    return parts
