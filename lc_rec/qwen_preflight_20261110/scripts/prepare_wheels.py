"""Prepare fixed official PyPI wheels for the remote CPython 3.10 overlays.

This is a CPU-only WSL preparation step. It never installs packages or imports
Torch, and the SSH inventory is read-only. Package payloads stay outside Git.
"""
import argparse
import datetime as dt
import email
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request
import zipfile

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import cpython_tags, compatible_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

TASK = Path(__file__).resolve().parents[1]
CONFIG = TASK / "config"
SUPPORT = Path("/home/addaswsw/recommend-support/qwen-preflight-20261110")
INVENTORY = CONFIG / "dependencies_remote_inventory.json"
NETWORK = CONFIG / "dependencies_network.json"
BASE = "/data/bwwu/envs/grid-20261009-e61baf3/bin/python"
FRAMEWORK = "/data/bwwu/envs/lc-rec-offload-20261108/lib/python3.10/site-packages"
LETTER = "/data/bwwu/envs/letter-20261110/lib/python3.10/site-packages"
PINS = {
    "qwen3": {"transformers": "4.57.6", "peft": "0.21.2", "accelerate": "1.12.0",
              "huggingface-hub": "0.36.2", "tokenizers": "0.22.2", "safetensors": "0.7.0"},
    "qwen35": {"transformers": "5.2.0", "peft": "0.21.2", "accelerate": "1.12.0",
               "huggingface-hub": "1.4.0", "tokenizers": "0.22.2", "safetensors": "0.7.0",
               "typer-slim": "0.21.1", "httpx": "0.28.1", "httpcore": "1.0.9",
               "anyio": "4.12.1", "h11": "0.16.0", "shellingham": "1.5.4",
               "exceptiongroup": "1.3.1"},
}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def event(row):
    records = json.loads(NETWORK.read_text()) if NETWORK.exists() else []
    records.append({"recorded_utc": utc(), **row})
    atomic(NETWORK, records)


def collect_inventory():
    code = '''import json,sys,platform,importlib.metadata as m,hashlib
from pathlib import Path
base_site='/data/bwwu/envs/grid-20261009-e61baf3/lib/python3.10/site-packages'
letter_site='/data/bwwu/envs/letter-20261110/lib/python3.10/site-packages'
sys.path.insert(sys.path.index(base_site),letter_site)
result={'python':sys.version,'executable':sys.executable,'sys_path':sys.path,
        'platform':platform.platform(),'libc':platform.libc_ver(),'packages':{}}
for d in m.distributions():
 name=d.metadata.get('Name')
 if name and name.lower().replace('_','-') not in result['packages']:
  result['packages'][name.lower().replace('_','-')]={'version':d.version,'root':str(d.locate_file('')),
      'requires_dist':d.requires or []}
print(json.dumps(result))
'''
    ssh = "/mnt/c/Windows/System32/OpenSSH/ssh.exe"
    interpreter = "/data/bwwu/envs/lc-rec-offload-20261108/bin/python"
    command = [ssh, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "new-server",
               shlex.join(["env", "CUDA_VISIBLE_DEVICES=", interpreter, "-B", "-"])]
    start = time.monotonic()
    result = subprocess.run(command, input=code, capture_output=True, text=True, timeout=60)
    receipt = {"argv": command, "seconds": time.monotonic() - start,
               "returncode": result.returncode, "stderr": result.stderr,
               "read_only": True, "CUDA_VISIBLE_DEVICES": "", "observed_utc": utc()}
    event({"kind": "ssh_distribution_inventory", **receipt})
    result.check_returncode()
    body = json.loads(result.stdout)
    assert body["python"].startswith("3.10.22"), body["python"]
    body["receipt"] = receipt
    atomic(INVENTORY, body)
    return body


def get_metadata(name, version, cache):
    key = (name, version)
    if key in cache:
        return cache[key]
    url = "https://pypi.org/pypi/{}/{}/json".format(name, version)
    errors = []
    for attempt in range(3):
        start = time.monotonic()
        try:
            with urllib.request.urlopen(url, timeout=40) as response:
                raw = response.read()
            data = json.loads(raw)
            assert canonicalize_name(data["info"]["name"]) == name
            assert data["info"]["version"] == version
            cache[key] = (data, hashlib.sha256(raw).hexdigest())
            event({"kind": "pypi_metadata", "url": url, "attempt": attempt,
                   "bytes": len(raw), "sha256": cache[key][1], "seconds": time.monotonic() - start,
                   "status": "PASS"})
            return cache[key]
        except Exception as error:
            errors.append(repr(error))
            event({"kind": "pypi_metadata", "url": url, "attempt": attempt,
                   "seconds": time.monotonic() - start, "status": "FAIL", "error": repr(error)})
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError("Official PyPI metadata failed: " + repr(errors))


def target_markers():
    env = default_environment()
    env.update(python_version="3.10", python_full_version="3.10.22", sys_platform="linux",
               platform_machine="x86_64", platform_system="Linux", os_name="posix", extra="")
    return env


def active_requirements(requires):
    env = target_markers()
    return [r for r in map(Requirement, requires or []) if r.marker is None or r.marker.evaluate(env)]


def select_wheel(data, inventory):
    libc = inventory["libc"]
    assert libc[0] == "glibc" and int(libc[1].split(".")[0]) == 2
    minor = int(libc[1].split(".")[1])
    platforms = ["manylinux_2_{}_x86_64".format(n) for n in range(minor, 16, -1)]
    platforms.extend(["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"])
    tags = list(cpython_tags(python_version=(3, 10), abis=["cp310"], platforms=platforms))
    tags += list(compatible_tags(python_version=(3, 10), interpreter="cp310", platforms=platforms))
    order = {tag: i for i, tag in enumerate(tags)}
    candidates = []
    for item in data["urls"]:
        if item["packagetype"] != "bdist_wheel" or item.get("yanked"):
            continue
        _, _, _, wheel_tags = parse_wheel_filename(item["filename"])
        scores = [order[t] for t in wheel_tags if t in order]
        if scores:
            candidates.append((min(scores), item["filename"], item))
    assert candidates, "No CPython 3.10 Linux x86_64 official wheel for " + data["info"]["name"]
    return min(candidates)[2]


def download(item, directory):
    path = directory / item["filename"]
    expected = item["digests"]["sha256"]
    assert item["url"].startswith("https://files.pythonhosted.org/")
    if path.exists():
        assert path.stat().st_size == item["size"] and digest(path) == expected
        event({"kind": "wheel_reuse", "path": str(path), "sha256": expected, "bytes": item["size"]})
        return path
    partial = path.with_name(path.name + ".partial")
    for attempt in range(3):
        start = time.monotonic()
        try:
            # Wheels are small (<20 MiB). Preserve a failed fragment for evidence;
            # a subsequent retry writes a separate file before atomic replace.
            tmp = partial.with_name(partial.name + ".attempt" + str(attempt))
            with urllib.request.urlopen(item["url"], timeout=60) as response, tmp.open("wb") as dest:
                while True:
                    block = response.read(1024 ** 2)
                    if not block:
                        break
                    dest.write(block)
            assert tmp.stat().st_size == item["size"] and digest(tmp) == expected
            tmp.replace(path)
            event({"kind": "wheel_download", "url": item["url"], "path": str(path), "attempt": attempt,
                   "bytes": item["size"], "sha256": expected, "seconds": time.monotonic() - start,
                   "status": "PASS", "installation": False})
            return path
        except Exception as error:
            event({"kind": "wheel_download", "url": item["url"], "attempt": attempt,
                   "seconds": time.monotonic() - start, "status": "FAIL", "error": repr(error)})
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError("Official wheel transfer failed: " + item["filename"])


def prepare(family, inventory, wheelhouse, cache):
    pins = PINS[family]
    inherited = {canonicalize_name(k): v for k, v in inventory["packages"].items()}
    pending = ["transformers", "peft", "accelerate"]
    selected, satisfied = {}, {}
    while pending:
        name = pending.pop(0)
        if name in selected:
            continue
        assert name in pins, "Missing explicit compatible dependency pin: " + name
        version = pins[name]
        data, metadata_sha = get_metadata(name, version, cache)
        assert Version("3.10.22") in SpecifierSet(data["info"]["requires_python"] or "")
        wheel = select_wheel(data, inventory)
        row = {"name": name, "version": version, "filename": wheel["filename"], "url": wheel["url"],
               "bytes": wheel["size"], "sha256": wheel["digests"]["sha256"],
               "requires_python": data["info"]["requires_python"],
               "requires_dist": data["info"]["requires_dist"], "pypi_metadata_sha256": metadata_sha,
               "source": "https://pypi.org/pypi/{}/{}/json".format(name, version),
               "reason": "fixed official model/PEFT compatibility in task-only overlay"}
        selected[name] = row
        for requirement in active_requirements(row["requires_dist"]):
            child = canonicalize_name(requirement.name)
            if child in pins:
                assert Version(pins[child]) in requirement.specifier, (name, str(requirement), pins[child])
                if child not in selected:
                    pending.append(child)
            else:
                current = inherited.get(child)
                assert current and Version(current["version"]) in requirement.specifier, (name, str(requirement), current)
                satisfied[child] = {**current, "reason": "existing dependency satisfies selected official metadata"}
    for row in selected.values():
        assert not row["name"].startswith(("torch", "nvidia", "triton", "deepspeed", "bitsandbytes"))
        data, _ = cache[(row["name"], row["version"])]
        wheel_path = download(select_wheel(data, inventory), wheelhouse)
        with zipfile.ZipFile(wheel_path) as archive:
            members = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            assert len(members) == 1
            raw = archive.read(members[0])
        metadata = email.message_from_bytes(raw)
        assert canonicalize_name(metadata["Name"]) == row["name"] and metadata["Version"] == row["version"]
        actual = metadata.get_all("Requires-Dist") or []
        assert {str(r) for r in active_requirements(actual)} == {str(r) for r in active_requirements(row["requires_dist"])}, row["name"]
        row["wheel_metadata_sha256"] = hashlib.sha256(raw).hexdigest()
        row["wheel_requires_dist"] = actual
    result = {"schema": "qwen_preflight_offline_dependencies_v1", "family": family,
              "target_python": "3.10.22", "target_platform": "Linux x86_64", "CUDA_required": False,
              "base_python": BASE, "readonly_framework_site": FRAMEWORK,
              "readonly_letter_site": LETTER,
              "environment": "/data/bwwu/envs/qwen-preflight-20261110-" + family,
              "packages": sorted(selected.values(), key=lambda r: r["name"]),
              "inherited_packages": satisfied, "inventory_sha256": digest(INVENTORY),
              "local_wheelhouse": str(wheelhouse), "payload_bytes": sum(x["bytes"] for x in selected.values()),
              "installation_status": "NOT STARTED", "prepared_utc": utc()}
    atomic(CONFIG / ("dependencies_" + family + ".json"), result)
    print(json.dumps({"family": family, "packages": len(selected), "bytes": result["payload_bytes"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", action="store_true", help="Refresh read-only remote distribution inventory")
    parser.add_argument("--family", choices=["qwen3", "qwen35", "all"], default="all")
    parser.add_argument("--wheelhouse", type=Path, default=SUPPORT / "wheels")
    args = parser.parse_args()
    assert not os.environ.get("CUDA_VISIBLE_DEVICES"), "This preparation uses no CUDA"
    inventory = collect_inventory() if args.inventory else json.loads(INVENTORY.read_text())
    assert args.wheelhouse.is_absolute()
    args.wheelhouse.mkdir(parents=True, exist_ok=True)
    cache = {}
    for family in (["qwen3", "qwen35"] if args.family == "all" else [args.family]):
        prepare(family, inventory, args.wheelhouse, cache)


if __name__ == "__main__":
    main()
