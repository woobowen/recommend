"""Fixed-release client for the authorized LC-Rec six-task Qwen preflight."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tarfile

REPO = Path(__file__).resolve().parents[3]
TASK = REPO / "lc_rec/qwen_six_task_20261111"
SUPPORT = Path("/home/addaswsw/recommend-support/lc-qwen-six-task-20261111")
REMOTE = "/home/bwwu/lc-rec/qwen-six-task-20261111"
DATA = "/data/bwwu/lc-rec/qwen-six-task-20261111"
PREFIX = "lc-qwen6-20261111-"
sys.path.insert(0, "/home/addaswsw/recommend-support/sid-extensions-p0/scripts")
import transport
import gridctl

transport.Transport.python = lambda self, code, timeout=45: self.ssh(
    shlex.join(["python3", "-I", "-S", "-B", "-c", code]), timeout=timeout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "deploy", "submit", "query", "logs", "collect", "snapshot"])
    parser.add_argument("--commit")
    parser.add_argument("--job")
    parser.add_argument("--spec")
    args = parser.parse_args()
    assert subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip() == "wbw-lc-rec"
    client = transport.Transport("new-server")
    SUPPORT.mkdir(parents=True, exist_ok=True)
    result = {}
    if args.mode == "preflight":
        code = """import json,subprocess,time,hashlib;from pathlib import Path
result={'observed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'memory':Path('/proc/meminfo').read_text(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv'],text=True),'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True),'disk':subprocess.check_output(['df','-B1','/home/bwwu','/data/bwwu'],text=True)}
result['runner_sha256']=hashlib.sha256(Path('/home/bwwu/grid-audits/sid-extensions-p0/scripts/runner-398768a732bbfebd.py').read_bytes()).hexdigest()
for name,revision in [('Qwen3-0.6B-Base','da87bfb608c14b7cf20ba1ce41287e8de496c0cd'),('Qwen3-1.7B-Base','ea980cb0a6c2ae4b936e82123acc929f1cec04c1')]:
 root=Path('/data/bwwu/lc-rec/qwen-preflight-20261110/models')/name/revision
 manifest=json.loads((root/'ASSET_MANIFEST.json').read_text())
 result[name]={'path':str(root),'prior_full_hash_receipt':manifest,'current_files':[]}
 for path in sorted(root.iterdir()):
  if not path.is_file():continue
  stat=path.stat();row={'name':path.name,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'inode':stat.st_ino}
  if stat.st_size<16*1024**2:row['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
  result[name]['current_files'].append(row)
print(json.dumps(result))"""
        result = json.loads(client.python(code))
        assert result["runner_sha256"] == client.profile["runner_sha256"]
        transport.atomic(TASK / "evidence/preflight.json", result)
    elif args.mode == "deploy":
        commit = subprocess.check_output(["git", "rev-parse", "--verify", (args.commit or "HEAD") + "^{commit}"], cwd=REPO, text=True).strip()
        archive = SUPPORT / (commit + ".tar.gz")
        subprocess.run(["git", "archive", "--format=tar.gz", "--output=" + str(archive), commit,
                        "lc_rec/qwen_six_task_20261111", "lc_rec/vendor/original",
                        "lc_rec/six_task_lora/scripts/data_support.py"], cwd=REPO, check=True)
        upload = client.profile["remote_support"] + "/transfers/lc-qwen6-" + commit + ".tar.gz"
        sent = client.put(archive, upload)
        release = REMOTE + "/releases/" + commit
        code = f"""import json,hashlib,tarfile;from pathlib import Path,PurePosixPath
root=Path({release!r});archive=Path({upload!r});assert hashlib.sha256(archive.read_bytes()).hexdigest()=={sent['sha256']!r}
rows=[]
with tarfile.open(archive) as stream:
 members=stream.getmembers();assert len(members)==len(set(member.name for member in members))
 for member in members:assert not PurePosixPath(member.name).is_absolute() and '..' not in PurePosixPath(member.name).parts and (member.isfile() or member.isdir())
 if not root.exists():root.mkdir(parents=True);stream.extractall(root,filter='data')
 for member in members:
  if member.isfile():
   content=stream.extractfile(member).read();path=root/member.name;assert not path.is_symlink() and path.read_bytes()==content
   rows.append({{'path':member.name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}})
result={{'RUN_COMMIT':{commit!r},'release':str(root),'not_a_git_checkout':True,'archive_sha256':{sent['sha256']!r},'files':rows}}
path=root/'RUN_VERSION.json'
if path.exists():assert json.loads(path.read_text())==result
else:path.write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result))"""
        result = json.loads(client.python(code, timeout=90))
        transport.atomic(SUPPORT / ("deployment-" + commit + ".json"), result)
        transport.atomic(TASK / "evidence/deployments" / (commit + ".json"), result)
    elif args.mode == "snapshot":
        code = f"""import json,subprocess,time,hashlib;from pathlib import Path
root=Path({REMOTE!r});result={{'observed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv'],text=True),'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True),'memory':Path('/proc/meminfo').read_text(),'disk':subprocess.check_output(['df','-B1','/home/bwwu','/data/bwwu'],text=True)}}
for name in ['gpu-budget.json','update-budget.json']:
 path=root/name
 if path.exists():result[name]=json.loads(path.read_text())
print(json.dumps(result))"""
        result = json.loads(client.python(code))
        transport.atomic(TASK / "evidence/final_remote.json", result)
    else:
        assert args.job and args.job.startswith(PREFIX)
        gridctl.valid(args.job)
        if args.mode == "submit":
            assert args.spec and args.commit
            commit = subprocess.check_output(["git", "rev-parse", "--verify", args.commit + "^{commit}"], cwd=REPO, text=True).strip()
            release = REMOTE + "/releases/" + commit
            task = release + "/lc_rec/qwen_six_task_20261111"
            jobdir = client.profile["job_root"] + "/" + args.job
            spec = json.loads(Path(args.spec).read_text())
            replacements = {"RELEASE": release, "TASK": task, "DATA": DATA, "REMOTE": REMOTE,
                            "JOB": jobdir, "COMMIT": commit}

            def expand(value):
                for key, replacement in replacements.items():
                    value = value.replace("{" + key + "}", replacement)
                return value

            profile = client.profile
            request = {"schema": 1, "host": profile["hostname"], "user": profile["user"],
                       "uid": profile["uid"], "namespace": profile["namespace"],
                       "deployment_sha256": profile["deployment_sha256"]}
            deployment = json.loads((SUPPORT / ("deployment-" + commit + ".json")).read_text())
            inputs = [{"path": release + "/" + row["path"], "sha256": row["sha256"]}
                      for row in deployment["files"] if "/scripts/" in row["path"] or "/config/" in row["path"] or "/vendor/original/" in row["path"]]
            request.update(job_id=args.job, capture_version=commit, cwd=release,
                           argv=[expand(value) for value in spec["argv"]],
                           env={key: expand(value) for key, value in spec["env"].items()},
                           inputs=inputs, timeout_seconds=spec["timeout_seconds"])
            path = SUPPORT / "requests" / (args.job + ".json")
            if path.exists():
                assert json.loads(path.read_text()) == request
            else:
                transport.atomic(path, request)
            result = gridctl.submit(client, path)
        elif args.mode in ["query", "logs"]:
            result = gridctl.query(client, args.job)
            if args.mode == "logs":
                print(client.ssh(shlex.join(["tail", "-n", "50", client.profile["job_root"] + "/" + args.job + "/work.log"])))
        else:
            result = gridctl.query(client, args.job)
            assert result["cleanup_verified"] and not result["worker_alive"] and not result["workgroup_live"]
            jobdir = client.profile["job_root"] + "/" + args.job
            destination = REMOTE + "/light_exports/" + args.job + ".tar.gz"
            code = f"""import json,hashlib,tarfile;from pathlib import Path
root=Path({jobdir!r});archive=Path({destination!r});rows=[];excluded=[]
for path in sorted(root.rglob('*')):
 if not path.is_file() or path.is_symlink() or path.name.endswith(('.lock','.partial','.tmp')):continue
 relative=str(path.relative_to(root))
 if path.stat().st_size>12*1024**2 or path.suffix in ['.bin','.safetensors','.pt','.pth','.gz'] or path.name in ['tokenizer.json','vocab.json','merges.txt']:
  excluded.append({{'path':relative,'bytes':path.stat().st_size}});continue
 if relative!='light-export-manifest.json':rows.append({{'path':relative,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}})
manifest=root/'light-export-manifest.json';manifest.write_text(json.dumps({{'files':rows,'excluded':excluded}},indent=2)+'\\n');archive.parent.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive,'w:gz') as stream:
 for row in rows:stream.add(root/row['path'],arcname=row['path'],recursive=False)
 stream.add(manifest,arcname=manifest.name,recursive=False)
print(json.dumps({{'path':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}}))"""
            metadata = json.loads(client.python(code, timeout=90))
            local = SUPPORT / (args.job + ".tar.gz")
            client.get_verified(metadata["path"], local, metadata, timeout=180)
            destination = TASK / "evidence/jobs" / args.job
            destination.mkdir(parents=True, exist_ok=True)
            with tarfile.open(local) as stream:
                names = [member.name for member in stream.getmembers()]
                assert len(names) == len(set(names))
                for member in stream.getmembers():
                    assert member.isfile() and not PurePosixPath(member.name).is_absolute() and ".." not in PurePosixPath(member.name).parts
                stream.extractall(destination, filter="data")
            manifest = json.loads((destination / "light-export-manifest.json").read_text())
            assert set(names) == {row["path"] for row in manifest["files"]} | {"light-export-manifest.json"}
            for row in manifest["files"]:
                path = destination / row["path"]
                assert path.stat().st_size == row["bytes"] and transport.sha(path) == row["sha256"]
            transport.atomic(destination / "controller-query.json", result)
        transport.atomic(SUPPORT / (args.mode + "-" + args.job + ".json"), result)
    keys = ["RUN_COMMIT", "release", "not_a_git_checkout", "archive_sha256", "job_id", "status", "exit_code", "cleanup_verified", "worker_alive", "supervisor_alive", "workgroup_live", "request_sha256", "submission", "observed_utc", "runner_sha256"]
    print(json.dumps({key: result[key] for key in keys if key in result}, indent=2))


if __name__ == "__main__":
    main()
