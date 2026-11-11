"""Read back fixed server releases and terminal jobs after the delivery commit."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
TASK = REPO / "lc_rec/qwen_six_task_20261111"
REMOTE = "/home/bwwu/lc-rec/qwen-six-task-20261111"
DATA = "/data/bwwu/lc-rec/qwen-six-task-20261111"
SUPPORT = Path("/home/addaswsw/recommend-support/lc-qwen-six-task-20261111")
sys.path.insert(0, "/home/addaswsw/recommend-support/sid-extensions-p0/scripts")
import transport
import gridctl

transport.Transport.python = lambda self, code, timeout=45: self.ssh(
    shlex.join(["python3", "-I", "-S", "-B", "-c", code]), timeout=timeout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    delivery = subprocess.check_output(["git", "rev-parse", "--verify", args.commit + "^{commit}"], cwd=REPO, text=True).strip()
    args.output.mkdir(parents=True, exist_ok=True)
    client = transport.Transport("new-server")
    jobs = sorted(path.name for path in (TASK / "evidence/jobs").iterdir() if path.is_dir())
    statuses = {}
    mappings = []
    commits = set()
    for job in jobs:
        assert job.startswith("lc-qwen6-20261111-")
        status = gridctl.query(client, job)
        assert status["cleanup_verified"] and not any(status[name] for name in ["worker_alive", "supervisor_alive", "workgroup_live"]), job
        statuses[job] = status
        request = json.loads((TASK / "evidence/jobs" / job / "request.json").read_text())
        commit = request["capture_version"]
        assert len(commit) == 40
        subprocess.check_call(["git", "cat-file", "-e", commit + "^{commit}"], cwd=REPO)
        release = REMOTE + "/releases/" + commit
        assert request["cwd"] == release
        commits.add(commit)
        files = []
        for row in request["inputs"]:
            assert row["path"].startswith(release + "/")
            relative = row["path"][len(release) + 1:]
            content = subprocess.check_output(["git", "show", commit + ":" + relative], cwd=REPO)
            assert hashlib.sha256(content).hexdigest() == row["sha256"]
            files.append({"path": relative, "sha256": row["sha256"], "bytes": len(content)})
        changed = subprocess.check_output(["git", "diff", "--name-only", commit, delivery, "--", "lc_rec/qwen_six_task_20261111/scripts", "lc_rec/qwen_six_task_20261111/config"], cwd=REPO, text=True).splitlines()
        gpu = "resource_guard.py" in " ".join(request["argv"])
        critical = ["scripts/common.py", "scripts/probe.py", "scripts/resource_guard.py"]
        mappings.append({"job_id": job, "run_commit": commit, "delivery_commit": delivery, "is_gpu_job": gpu,
                         "release": release, "server_export_is_not_git_checkout": True,
                         "request_source_bytes_match_git": True, "input_files": files,
                         "scripts_or_configs_changed_after_run": changed,
                         "gpu_implementation_unchanged_in_delivery": all("lc_rec/qwen_six_task_20261111/" + path not in changed for path in critical) if gpu else None})
    releases = []
    for commit in sorted(commits):
        manifest = json.loads((SUPPORT / ("deployment-" + commit + ".json")).read_text())
        releases.append({"commit": commit, "release": manifest["release"], "files": [row for row in manifest["files"] if "/scripts/" in row["path"] or "/config/" in row["path"] or "/vendor/original/" in row["path"]]})
    code = f"""import hashlib,json,os,shutil,subprocess,time;from pathlib import Path
releases={[{key: release[key] for key in ['commit', 'release']} for release in releases]!r};results=[]
for release in releases:
 root=Path(release['release']);version=json.loads((root/'RUN_VERSION.json').read_text());assert version['RUN_COMMIT']==release['commit'] and version['not_a_git_checkout'] is True and not (root/'.git').exists()
 rows=[]
 for expected in version['files']:
  if not ('/scripts/' in expected['path'] or '/config/' in expected['path'] or '/vendor/original/' in expected['path']):continue
  path=root/expected['path'];assert path.is_file() and not path.is_symlink();content=path.read_bytes();actual={{'path':expected['path'],'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}};assert actual==expected;rows.append(actual)
 results.append({{'run_commit':release['commit'],'release':str(root),'files':rows,'all_bytes_match':True,'not_a_git_checkout':True}})
jobroot=Path({client.profile['job_root']!r});roots=[Path({REMOTE!r}),Path({DATA!r})]+list(jobroot.glob('lc-qwen6-20261111-*'))
logical=allocated=0;root_sizes=[]
for root in roots:
 root_logical=root_allocated=0
 for directory,directories,files in os.walk(root,followlinks=False):
  directories[:]=[name for name in directories if not (Path(directory)/name).is_symlink()]
  for name in files:
   path=Path(directory)/name
   if path.is_symlink():continue
   stat=path.stat();root_logical+=stat.st_size;root_allocated+=stat.st_blocks*512
 logical+=root_logical;allocated+=root_allocated;root_sizes.append({{'root':str(root),'logical_bytes':root_logical,'allocated_bytes':root_allocated}})
for folder,pattern in [('transfers','lc-qwen6-*'),('requests','lc-qwen6-20261111-*')]:
 for path in (jobroot.parent/folder).glob(pattern):
  if path.is_file() and not path.is_symlink():
   stat=path.stat();logical+=stat.st_size;allocated+=stat.st_blocks*512
budget={{}}
for name in ['gpu-budget.json','update-budget.json']:
 path=Path({REMOTE!r})/name
 if path.exists():budget[name]=json.loads(path.read_text())
result={{'observed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'releases':results,'task_disk':{{'logical_bytes':logical,'allocated_bytes':allocated,'roots':root_sizes,'limit_bytes':20*1024**3}},'budgets':budget,'memory':Path('/proc/meminfo').read_text(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv'],text=True),'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True),'disk':subprocess.check_output(['df','-B1','/home/bwwu','/data/bwwu'],text=True),'runner_sha256':hashlib.sha256(Path({client.profile['runner']!r}).read_bytes()).hexdigest()}}
assert allocated<20*1024**3
for location in ['/home/bwwu','/data/bwwu']:
 disk=shutil.disk_usage(location);assert disk.free>=max(30*1024**3,disk.total//10)
print(json.dumps(result))"""
    readback = json.loads(client.python(code, timeout=90))
    expected_releases = {release["commit"]: release for release in releases}
    assert len(readback["releases"]) == len(expected_releases)
    for observed in readback["releases"]:
        expected = expected_releases[observed["run_commit"]]
        assert observed["release"] == expected["release"]
        assert observed["files"] == expected["files"]
    assert readback["runner_sha256"] == client.profile["runner_sha256"]
    assert not readback["gpu_processes"].strip(), "GPU occupancy changed before final receipt"
    budgets = readback["budgets"]
    ledger = budgets.get("gpu-budget.json", {"runs": []})
    assert all(row.get("end_utc") and row.get("cleanup_verified") for row in ledger["runs"])
    seconds = sum(row["charged_gpu_device_seconds"] for row in ledger["runs"])
    assert seconds <= 14400
    per_model = {key: sum(row["charged_gpu_device_seconds"] for row in ledger["runs"] if row["key"] == key) for key in ["q06", "q17"]}
    assert all(seconds <= 7200 for seconds in per_model.values())
    update_ledger = budgets.get("update-budget.json", {"reservations": []})
    reservations = update_ledger["reservations"]
    assert len(reservations) <= 14
    for route, limit in {"q06-L": 4, "q06-F": 3, "q17-L": 4, "q17-F": 3}.items():
        assert sum(row["route_key"] == route for row in reservations) <= limit
    readback["verified_budget"] = {"total_gpu_device_seconds": seconds, "per_model_device_seconds": per_model, "charged_optimizer_boundaries": len(reservations)}
    readback["receipt_controller"] = {"script": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                      "argv": sys.argv, "python": sys.version, "delivery_commit": delivery,
                                      "transport": "Existing Windows OpenSSH identity; compact release paths with remote RUN_VERSION file reads and complete local manifest comparison"}
    transport.atomic(args.output / "server_readback.json", readback)
    transport.atomic(args.output / "terminal_jobs.json", statuses)
    transport.atomic(args.output / "run_delivery_mapping.json", {"delivery_commit": delivery, "jobs": mappings, "all_server_input_hashes_and_git_objects_match": True})
    transport.atomic(args.output / "installation_receipt.json", {"system_packages": [], "language_packages": [], "toolchains": [], "shared_environment_changes": [], "global_config_changes": [], "reused_environment": "/data/bwwu/envs/qwen-preflight-20261110-qwen3", "task_owned_changes": "Source/config/results, task-local GPU/runtime caches and model/checkpoint outputs only; no cleanup performed"})
    print(json.dumps({"delivery_commit": delivery, "terminal_jobs": len(jobs), "server_releases_read_back": len(releases), "total_gpu_device_minutes": seconds / 60, "new_disk_allocated_bytes": readback["task_disk"]["allocated_bytes"], "all_task_workers_cleaned": True}, indent=2))


if __name__ == "__main__":
    main()
