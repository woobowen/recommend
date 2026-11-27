#!/usr/bin/env python3
"""Read published GitHub bytes by fixed commit and compare them with Git objects."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

BRANCH = "wbw-lc-rec-shuffle-v2"
REPO = "woobowen/recommend"


def check(args):
    ref_result = subprocess.check_output(["git", "ls-remote", "origin", f"refs/heads/{BRANCH}"], text=True)
    live = ref_result.split()[0]
    if live != args.commit:
        raise ValueError(f"GitHub ref is {live}, exact expected commit is {args.commit}")

    def compare(path):
        location = f"repos/{REPO}/contents/{path}?ref={args.commit}"
        remote = subprocess.check_output(["gh", "api", "-H", "Accept: application/vnd.github.raw+json", location])
        local = subprocess.check_output(["git", "show", f"{args.commit}:{path}"])
        if remote != local:
            raise ValueError(f"GitHub readback bytes differ for {path}")
        return {"path": path, "bytes": len(remote), "sha256": hashlib.sha256(remote).hexdigest()}

    with ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(compare, args.files))
    payload = {"verify_utc": datetime.now(timezone.utc).isoformat(), "repo": REPO,
               "branch": BRANCH, "live_head": live, "readback": files}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"Verified GitHub {live} and {len(files)} actual file payloads")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("files", nargs="+")
    check(parser.parse_args())
