"""Verify the GitHub ref and contents of this fast-forward publication."""
import base64
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

repo = Path(sys.argv[1])
commit = sys.argv[2]
branch = "wbw-grid"
base = "repos/woobowen/recommend"
experiment = "experiments/letter_instruments_grid_normal"
stage = experiment + "/evidence/formal-launch-30min"

def api(endpoint):
    return json.loads(subprocess.check_output(["gh", "api", base + "/" + endpoint], timeout=30))

def git(*args):
    return subprocess.check_output(["git", "-C", str(repo), *args])

local = {}
for raw in git("ls-tree", "-rz", commit, "--", f"{experiment}/README.md", stage).split(b"\0"):
    if not raw:
        continue
    header, path = raw.decode().split("\t", 1)
    mode, kind, sha = header.split()
    assert kind == "blob" and mode == "100644", (mode, kind, path)
    local[path] = sha
assert f"{stage}/snapshots/T30.json" in local

ref = api(f"git/ref/heads/{branch}")
assert ref["object"]["sha"] == commit
tree = api(f"git/trees/{commit}?recursive=1")
assert not tree["truncated"]
remote = {x["path"]:x["sha"] for x in tree["tree"] if x["type"] == "blob"}
assert {p:remote[p] for p in local} == local
critical = [
    f"{experiment}/README.md", f"{stage}/README.md", f"{stage}/server-request.json",
    f"{stage}/snapshots/T00.json", f"{stage}/snapshots/T30.json",
    f"{stage}/captured/output/run/effective_config.yaml", f"{stage}/NON_AUTHOR_REVIEW.md",
]
def read_blob(path):
    item = api("git/blobs/" + local[path])
    assert item["encoding"] == "base64"
    content = base64.b64decode(item["content"])
    assert content == git("show", commit + ":" + path)
    return dict(path=path, blob=local[path], bytes=len(content), sha256=hashlib.sha256(content).hexdigest())

with ThreadPoolExecutor(max_workers=4) as pool:
    reads = list(pool.map(read_blob, critical))
print(json.dumps(dict(
    repository="woobowen/recommend", branch=branch, commit=commit, api_ref=ref["ref"],
    read_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    compared_tree_entries=local, directly_read_blobs=reads, tree_truncated=False,
), indent=2))
