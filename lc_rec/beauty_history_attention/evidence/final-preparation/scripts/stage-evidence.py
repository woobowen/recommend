"""Copy selected readable final-preparation evidence into this experiment's existing evidence tree."""

import shutil
from pathlib import Path


SUPPORT = Path("/home/addaswsw/recommend-support/beauty-history-attention-final-20261127")
REVIEW = SUPPORT.parent / "letter-beauty-history-attention-formal-prep-independent-20261127"
REPO = Path("/home/addaswsw/recommend-worktrees/shuffle-v2")
TARGET = REPO / "lc_rec/beauty_history_attention/evidence/final-preparation"


def put(source, rel):
    output = TARGET / rel
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)


def copy(folder, prefix, allowed):
    for p in sorted(folder.rglob("*")):
        if p.is_file() and not p.is_symlink() and p.suffix in allowed and "__pycache__" not in p.parts:
            put(p, prefix / p.relative_to(folder))


for folder in sorted((SUPPORT / "collected").glob("*")):
    if folder.is_dir() and folder.name.startswith("letter-beauty-v2-history-attn-final-"):
        for member in ("request.json", "status.json", "cleanup.json", "accepted.json", "attempt.json",
                       "work.log", "supervisor.log"):
            path = folder / member
            if path.is_file():
                put(path, Path("runner", folder.name, member))
    elif folder.is_dir():
        copy(folder, Path("actual", folder.name), {".json", ".jsonl", ".log", ".txt"})
copy(SUPPORT / "evidence/formal", Path("formal"), {".json", ".txt", ".log", ".sh"})
copy(REVIEW, Path("nonauthor"), {".py", ".sh", ".json", ".txt", ".log", ".md"})
for name in ("resources.json", "combined-cpu-tests.log", "prediction-store-tests.log",
             "generation-store-cpu.log", "runtime-dependency-origins.json",
             "first-chunk-before-resume.sha256", "first-chunk-after-resume.sha256",
             "first-chunk-after-duplicate.sha256"):
    file = SUPPORT / "evidence" / name
    if file.is_file():
        put(file, Path("verification", name))
for ver in ("5a1a7297aa4078b46aa800f70ab15a649304cd67",
            "d43787a43a64c175ff6edbf3d1820e5fd0c2baa2",
            "81d4dc00f84426c77b48b7beb57b57903e97c20d"):
    put(SUPPORT / f"export-{ver}/SOURCE.sha256", Path("deployment", ver, "SOURCE.sha256"))
    put(SUPPORT / f"evidence/deploy-{ver}.log", Path("deployment", ver, "remote-verification.log"))
for name in ("deploy.sh", "submit.sh", "query.sh", "collect.sh", "pilot-sequence.sh",
             "resource-estimate.py", "build-review-zip.py", "stage-evidence.py", "readback.py"):
    put(SUPPORT / name, Path("scripts", name))
