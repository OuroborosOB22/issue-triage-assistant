import json
import re
import urllib.request
from pathlib import Path

TITLE = re.compile(r"^(?:chore\(deps\):\s*|build\(deps\):\s*)?(?:bump|update|upgrade)\s+(?P<name>\S+)\s+from\s+(?P<before>\S+)\s+to\s+(?P<after>\S+)\s*$", re.I)
API = "https://api.github.com"
PAGE_SIZE = 50


def parse_title(title):
    match = TITLE.match(title.strip())
    return match.groupdict() if match else None


def single_dependency(files):
    names = [item["filename"] for item in files]
    return len(names) == 1 and names[0].split("/")[-1] == "pom.xml"


def github_get(path, token=None):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "breakcast-candidates"}
    if token:
        headers["Authorization"] = "Bearer " + token
    with urllib.request.urlopen(urllib.request.Request(API + path, headers=headers), timeout=30) as response:
        return json.loads(response.read())


def candidates_for(repo, get=github_get, pages=2):
    found = []
    for page in range(1, pages + 1):
        pulls = get("/repos/%s/pulls?state=closed&per_page=%d&page=%d" % (repo, PAGE_SIZE, page))
        if not pulls:
            break
        for pull in pulls:
            parsed = parse_title(pull["title"])
            if not parsed or not pull.get("merged_at"):
                continue
            files = get("/repos/%s/pulls/%d/files" % (repo, pull["number"]))
            if not single_dependency(files):
                continue
            found.append({"repo": repo, "pull_request": pull["html_url"], "merged_at": pull["merged_at"], "dependency": parsed["name"],
                          "from_version": parsed["before"], "to_version": parsed["after"], "base_commit": pull["base"]["sha"], "head_commit": pull["head"]["sha"]})
    return found


def write_manifest(candidates, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for item in candidates:
            handle.write(json.dumps({**item, "ecosystem": "MAVEN", "change": "version-only", "status": "candidate-not-verified"}, sort_keys=True) + "\n")
