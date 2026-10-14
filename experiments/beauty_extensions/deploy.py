"""Use the accepted Windows transport/runner to deploy a hash-bound small snapshot.

This is local orchestration only; it never installs anything or changes the old
export. Each snapshot is a fresh copy of source bytes, without mutable hardlinks.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

from .common import REPO, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--support", type=Path, default=Path("/home/addaswsw/recommend-support/sid-extensions-p0"))
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--phase", required=True)
    p.add_argument("--entrypoint", choices=("cli", "mlp_execution", "mlp_formal", "mlp_attention", "plot_training"), default="cli")
    p.add_argument("--job", required=True)
    p.add_argument("--timeout", type=int, default=1200)
    args = p.parse_args()
    sys.path.insert(0, str(args.support / "scripts"))
    from transport import Transport, atomic
    import gridctl
    t = Transport("new-server")
    t.identity()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    assert branch == "wbw", "Only the authorized daily branch may be deployed"
    files = []
    for directory in ("GRID baseline/src", "references/wmh-original/GRID/src", "experiments/beauty_extensions"):
        for f in (REPO / directory).rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts and f.suffix in (".py", ".json", ".yaml"):
                files.append(f)
    for marker in ("GRID baseline/.project-root", "references/wmh-original/GRID/.project-root"):
        if (REPO / marker).is_file():
            files.append(REPO / marker)
    records = [{"path": str(f.relative_to(REPO)), "sha256": sha256(f), "bytes": f.stat().st_size}
               for f in sorted(files)]
    committed = args.phase in ("formal", "capture")
    if committed:
        for record in records:
            data = subprocess.check_output(["git", "cat-file", "blob", head + ":" + record["path"]], cwd=REPO)
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise RuntimeError("Formal deployment contains bytes outside the fixed commit: " + record["path"])
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    deployment = {"source_commit": head, "snapshot_sha256": digest, "files": records,
                  "identity_kind": "git_commit_and_exact_file_manifest" if committed else "source_commit_plus_exact_file_manifest; may contain preflight patch"}
    local = args.support / "beauty-extensions"
    local.mkdir(exist_ok=True)
    archive = local / ("snapshot-" + digest[:16] + ".tar.gz")
    if not archive.exists():
        with tarfile.open(archive, "w:gz") as tf:
            for f in sorted(files):
                tf.add(f, arcname=str(f.relative_to(REPO)), recursive=False)
            data = (json.dumps(deployment, indent=2) + "\n").encode()
            info = tarfile.TarInfo("DEPLOYMENT.json"); info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    remote_base = t.profile["remote_support"] + "/beauty-extensions"
    remote_archive = remote_base + "/archives/" + archive.name
    t.put(archive, remote_archive)
    remote_snapshot = remote_base + "/snapshots/" + digest[:16]
    # Saved, deterministic deployment operation; no source mutations on server.
    code = """from pathlib import Path
import tarfile,hashlib,json,os
archive=Path(ARCHIVE);root=Path(TARGET)
if not root.exists():
 root.mkdir(parents=True)
 with tarfile.open(archive,'r:gz') as tf:
  members=tf.getmembers()
  assert len({m.name for m in members})==len(members)
  assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
  tf.extractall(root,filter='data')
manifest=json.loads((root/'DEPLOYMENT.json').read_text())
assert manifest['snapshot_sha256']==DIGEST
for r in manifest['files']:
 p=root/r['path'];assert p.is_file() and not p.is_symlink()
 assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
print(json.dumps({'snapshot':str(root),'file_count':len(manifest['files']),'verified':True,'manifest':manifest['snapshot_sha256']}))
""".replace("ARCHIVE", repr(remote_archive)).replace("TARGET", repr(remote_snapshot)).replace("DIGEST", repr(digest))
    deployed = json.loads(t.python(code, timeout=60))
    atomic(local / (digest[:16] + "-deployment.json"), deployed)
    spec = json.loads(args.spec.read_text())
    remote_spec = remote_base + "/specs/" + sha256(args.spec)[:16] + ".json"
    t.put(args.spec, remote_spec)
    # This run's GPU is the physically verified idle second card.
    request = {"schema": 1, "job_id": args.job, "host": t.profile["hostname"],
        "user": t.profile["user"], "uid": t.profile["uid"], "namespace": t.profile["namespace"],
        "deployment_sha256": t.profile["deployment_sha256"], "capture_version": head + "/" + digest,
        "argv": ["/data/bwwu/envs/grid-20261009-e61baf3/bin/python", "-B", "-m",
                 "experiments.beauty_extensions." + args.entrypoint, args.phase, "--spec", remote_spec],
        "cwd": remote_snapshot, "timeout_seconds": args.timeout,
        "env": {"CUDA_VISIBLE_DEVICES": "" if args.entrypoint == "plot_training" else "GPU-59cac3bd-3645-668b-79c8-839e0d878893", "OMP_NUM_THREADS": "4",
                "TF_CPP_MIN_LOG_LEVEL": "3", "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "2",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": remote_snapshot, "HF_HUB_OFFLINE": "1",
                "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false"},
        "inputs": [{"path": remote_spec, "sha256": sha256(args.spec)},
                   {"path": remote_snapshot + "/DEPLOYMENT.json", "sha256": hashlib.sha256((json.dumps(deployment, indent=2) + "\n").encode()).hexdigest()}]}
    request_path = local / (args.job + "-request.json")
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise RuntimeError("Job identity exists with different inputs")
    write_json(request_path, request)
    receipt = gridctl.submit(t, request_path)
    print(json.dumps({"deployment": deployed, "receipt": receipt}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
