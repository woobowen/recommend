"""Bounded formal-stage evidence collection; reuse accepted collector, exclude weights/data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def collect_remote(plan):
    root=Path(plan["job_root"])
    assert str(root)=="/home/bwwu/grid-audits/sid-extensions-p0/jobs"
    target=Path(plan["target"])
    assert str(target).startswith("/home/bwwu/grid-audits/sid-extensions-p0/grid-formal-20261015/evidence/")
    assert not target.exists()
    target.parent.mkdir(parents=True,exist_ok=True)
    members=[];total=0
    for job in plan["jobs"]:
        assert job.startswith("gf-20261015-") and "/" not in job and ".." not in job
        base=root/job
        status=json.loads((base/"status.json").read_text())
        assert status["status"] in ("SUCCEEDED","FAILED","TIMED_OUT","CANCELLED")
        selected=[base/name for name in ("request.json","status.json","accepted.json","attempt.json","cleanup.json","work.log","supervisor.log","export-manifest.json") if (base/name).is_file()]
        for p in (base/"output").rglob("*"):
            if p.is_file() and not p.is_symlink() and "checkpoints" not in p.relative_to(base).parts and p.suffix in (".json",".jsonl",".yaml",".csv",".npz",".png"):
                assert p.stat().st_size<=64*1024**2
                selected.append(p)
        for p in sorted(set(selected)):
            assert not p.is_symlink()
            relative=job+"/"+str(p.relative_to(base))
            record={"path":relative,"bytes":p.stat().st_size,"sha256":sha(p)}
            total+=record["bytes"];assert total<128*1024**2
            members.append((p,record))
    manifest={"jobs":plan["jobs"],"files":[r for _,r in members],"uncompressed_bytes":total,
              "checkpoints_excluded":True,"source_data_excluded":True}
    manifest_path=target.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    with tarfile.open(target,"w:gz") as tf:
        for path,record in members:tf.add(path,arcname=record["path"],recursive=False)
        tf.add(manifest_path,arcname="COLLECTION_MANIFEST.json",recursive=False)
    return {"path":str(target),"bytes":target.stat().st_size,"sha256":sha(target),"files":len(members)}


def local(support, jobs):
    sys.path.insert(0,str(support/"scripts"))
    from transport import Transport,atomic
    import gridctl
    t=Transport("new-server");states={}
    for job in jobs:
        r=gridctl.query(t,job)
        assert r["status"] in ("SUCCEEDED","FAILED","TIMED_OUT","CANCELLED") and r["cleanup_verified"] and not r["workgroup_live"]
        states[job]={k:r.get(k) for k in ("status","exit_code","started_at","ended_at","result_path","worker_alive","cleanup_verified","request_sha256")}
    out=Path("/home/addaswsw/recommend-support/grid-formal-20261015");out.mkdir(exist_ok=True)
    tag=hashlib.sha256(json.dumps(jobs).encode()).hexdigest()[:16]
    plan={"job_root":t.profile["job_root"],"jobs":jobs,
          "target":t.profile["remote_support"]+"/grid-formal-20261015/evidence/collection-"+tag+".tar.gz"}
    lp=out/("collect-"+tag+".json");atomic(lp,plan)
    source=Path(__file__).resolve();rs=t.profile["remote_support"]+"/grid-formal-20261015/scripts/collect-"+sha(source)[:16]+".py"
    rp=t.profile["remote_support"]+"/grid-formal-20261015/specs/collect-"+sha(lp)[:16]+".json"
    t.put(source,rs);t.put(lp,rp)
    import shlex
    meta=json.loads(t.ssh(shlex.join(["/usr/bin/python3","-I","-S","-B",rs,"remote",str(rp)]),timeout=60))
    archive=out/("collected-"+tag+".tar.gz");receipt=t.get_verified(meta["path"],archive,meta)
    dest=out/("collected-"+tag);dest.mkdir(exist_ok=False)
    with tarfile.open(archive,"r:gz") as tf:
        members=tf.getmembers();names=[m.name for m in members]
        assert len(names)==len(set(names))
        assert all(m.isfile() and not Path(m.name).is_absolute() and ".." not in Path(m.name).parts for m in members)
        tf.extractall(dest,filter="data")
    manifest=json.loads((dest/"COLLECTION_MANIFEST.json").read_text())
    assert set(names)=={r["path"] for r in manifest["files"]}|{"COLLECTION_MANIFEST.json"}
    for r in manifest["files"]:
        f=dest/r["path"];assert f.stat().st_size==r["bytes"] and sha(f)==r["sha256"]
    atomic(out/"terminal-states.json",states)
    atomic(out/"collection-receipt.json",dict(receipt,directory=str(dest),members_verified=True))
    print(json.dumps({"receipt":receipt,"directory":str(dest),"terminal_states":states},indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="mode",required=True)
    r=sub.add_parser("remote");r.add_argument("plan",type=Path)
    l=sub.add_parser("local");l.add_argument("--support",type=Path,default=Path("/home/addaswsw/recommend-support/sid-extensions-p0"));l.add_argument("jobs",nargs="+")
    a=p.parse_args()
    if a.mode=="remote":print(json.dumps(collect_remote(json.loads(a.plan.read_text())),indent=2))
    else:local(a.support,a.jobs)
