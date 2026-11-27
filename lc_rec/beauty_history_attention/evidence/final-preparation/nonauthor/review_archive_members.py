"""Independent ZIP member, digest, source and original dependency review."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import zipfile

parser=argparse.ArgumentParser()
parser.add_argument("archive",type=Path)
parser.add_argument("--repo",type=Path,required=True)
parser.add_argument("--source-commit",required=True)
args=parser.parse_args()

def digest(data): return hashlib.sha256(data).hexdigest()

external=json.loads((Path(__file__).parent/"dependency_inventory_review.json").read_text())
need = {label:item["sha256"] for label,item in external.items()}
need["pinned_native_llama"]= "2abf84a5dd3644330f93c88e036831578e407de7a9812b5454ef1bb8a18c3422"
base="lc_rec/beauty_history_attention/"
required_modules = (
    "attention.py","identity.py","inputs.py","inference_attention.py","inference_probe.py","prediction_store.py",
    "evaluate.py","formal.py","train.py","final_preflight.py","config_bidirectional.json","historical_train_config.json",
)
for name in required_modules:
    data=subprocess.check_output(["git","-C",str(args.repo),"show",args.source_commit+":"+base+name])
    need[name]=digest(data)
source_dir = Path(__file__).parent
for name in ("FINAL.md", "ADDENDUM.md", "real_eager_independent.json", "real_resumable_generation_audit.json",
             "preflight_namespace_review-fixed.json"):
    need[name] = digest((source_dir / name).read_bytes())
by_digest=defaultdict(list)
with zipfile.ZipFile(args.archive) as z:
    members=z.infolist()
    names=[entry.filename for entry in members]
    assert len(set(names))==len(names), "duplicate member names"
    assert len(set(n.casefold() for n in names))==len(names), "case-collision in Windows"
    for info in members:
        path=PurePosixPath(info.filename)
        perms=info.external_attr>>16
        assert not path.is_absolute() and ".." not in path.parts and "\\" not in info.filename
        assert not stat.S_ISLNK(perms), f"symlink {info.filename}"
        if not info.is_dir():
            assert path.suffix.lower() not in (".zip",".tar",".gz"), f"nested archive {info.filename}"
            h=hashlib.sha256()
            with z.open(info) as stream:  # complete read also checks every member's CRC
                while block:=stream.read(1<<20):
                    h.update(block)
            by_digest[h.hexdigest()].append(info.filename)
    manifest = json.loads(z.read("MEMBERS.json"))
    assert manifest["formal_snapshot"] == args.source_commit, "archive declares a different formal snapshot"
    for name, entry in manifest["files"].items():
        source = entry.get("duplicate_of", name)
        assert source in by_digest[entry["sha256"]] and z.getinfo(source).file_size == entry["bytes"]
    final_plan = manifest["files"]["transport/formal-final/b-formal-plan.unsigned.json"]
    assert final_plan["sha256"] == "664694b681b92c9c46c90aa85c4123467233e86c31f087fa74d2b349e9ee9091"
    gate = manifest["files"]["transport/formal-final/gate-results.txt"]
    gate_text = z.read(gate.get("duplicate_of", "transport/formal-final/gate-results.txt")).decode()
    gate_result, _ = json.JSONDecoder().raw_decode(gate_text)
    assert not gate_result["formal_run_started"] and len(gate_result["negative_exit_codes"]) == 11
    found={label:by_digest[hash] for label,hash in need.items()}
    missing=[name for name,entry in found.items() if not entry]
    assert not missing, f"missing exact archive bytes: {missing}"
result=dict(status="PASS",archive=str(args.archive),bytes=args.archive.stat().st_size,
            archive_sha256=digest(args.archive.read_bytes()),members=len(members),
            marked_source=args.source_commit,required_byte_matches=found)
print(json.dumps(result,indent=2))
