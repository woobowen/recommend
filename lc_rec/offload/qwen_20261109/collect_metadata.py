#!/usr/bin/env python3
"""Read-only official Qwen metadata preparation; never fetches model tensors."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
MAX_TOTAL = 30_000_000
REPOS = ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B-Base")
SMALL = ("README.md", "LICENSE", "config.json", "generation_config.json",
         "tokenizer_config.json", "model.safetensors.index.json")
EVENTS = json.loads((ROOT / "fetch_events.json").read_text()) if (ROOT / "fetch_events.json").exists() else []


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def fetch(url, path, limit=2_000_000):
    if path.exists():
        data = path.read_bytes()
        if len(data) > limit:
            raise RuntimeError("Existing file exceeds size bound")
        return data
    for attempt in range(2):
        start = dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LC-Rec-Qwen-identity-audit/1.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read(limit + 1)
                if len(data) > limit:
                    raise RuntimeError("Response exceeds size bound")
                total = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
                if total + len(data) > MAX_TOTAL:
                    raise RuntimeError("Preparation total would exceed 30 MB")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                EVENTS.append({"time_utc": start, "url": url, "status": response.status,
                               "bytes": len(data), "path": str(path.relative_to(ROOT)),
                               "sha256": hashlib.sha256(data).hexdigest(),
                               "attempt": attempt})
                write_json(ROOT / "fetch_events.json", EVENTS)
                return data
        except Exception as exc:
            EVENTS.append({"time_utc": start, "url": url, "attempt": attempt,
                           "error": type(exc).__name__ + ": " + str(exc)})
            write_json(ROOT / "fetch_events.json", EVENTS)
            if attempt == 1:
                raise
            time.sleep(1)
    raise AssertionError("Unreachable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-payload", action="store_true",
                        help="Fetch bounded tokenizer JSON/text files; share only identical blob IDs")
    args = parser.parse_args()
    summary = []
    for repo in REPOS:
        short = repo.rsplit("/", 1)[1]
        api = f"https://huggingface.co/api/models/{repo}?blobs=true"
        info = json.loads(fetch(api, ROOT / short / "model_info.json"))
        revision = info["sha"]
        assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
        siblings = {s["rfilename"]: s for s in info["siblings"]}
        for name in SMALL:
            if name not in siblings:
                continue
            fetch(f"https://huggingface.co/{repo}/resolve/{revision}/{name}",
                  ROOT / short / name)
        weights = []
        for name, entry in siblings.items():
            if name.endswith(".safetensors"):
                lfs = entry.get("lfs", {})
                weights.append({"filename": name, "bytes": entry.get("size", lfs.get("size")),
                                "sha256": lfs.get("sha256"), "git_blob_id": entry.get("blobId"),
                                "url": f"https://huggingface.co/{repo}/resolve/{revision}/{name}"})
        total = sum(w["bytes"] for w in weights)
        summary.append({"repo": repo, "revision": revision,
                        "weight_files": weights, "weight_bytes": total,
                        "weight_GiB": total / 2**30,
                        "downloads_of_weights_this_task": 0,
                        "selected_by_user": False,
                        "tokenizer_files": {n: siblings[n] for n in ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json") if n in siblings}})
    if args.tokenizer_payload:
        for name in ("tokenizer.json", "vocab.json", "merges.txt"):
            entries = [s["tokenizer_files"].get(name) for s in summary]
            if not all(entries):
                raise RuntimeError("Tokenizer file missing")
            shared = len({e.get("blobId") for e in entries}) == 1
            for i, (s, entry) in enumerate(zip(summary, entries)):
                if entry.get("size", 0) > 15_000_000:
                    raise RuntimeError("Tokenizer file too large")
                local = (Path("shared_tokenizer") if shared else Path(s["repo"].rsplit("/", 1)[1])) / name
                s.setdefault("tokenizer_local_paths", {})[name] = str(local)
                if i == 0 or not shared:
                    data = fetch(f"https://huggingface.co/{s['repo']}/resolve/{s['revision']}/{name}",
                                 ROOT / local, limit=15_000_000)
                    if "lfs" in entry:
                        valid = hashlib.sha256(data).hexdigest() == entry["lfs"]["sha256"]
                    else:
                        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                        valid = blob == entry["blobId"]
                    if not valid:
                        raise RuntimeError("Tokenizer source content identity mismatch")
    write_json(ROOT / "candidate_summary.json", summary)
    write_json(ROOT / "fetch_events.json", EVENTS)
    print(json.dumps([{k: s[k] for k in ("repo", "revision", "weight_bytes", "weight_GiB")} for s in summary], indent=2))


if __name__ == "__main__":
    main()
