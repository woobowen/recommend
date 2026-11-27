#!/usr/bin/env python3
"""Publish human-readable raw evidence with the locally retained assets omitted."""
import shutil
from pathlib import Path

SUPPORT = Path("/home/addaswsw/recommend-support/beauty-history-attention-20261127")
EVIDENCE = Path("/home/addaswsw/recommend-worktrees/shuffle-v2/lc_rec/beauty_history_attention/evidence")
INDEPENDENT = SUPPORT.parent / "letter-beauty-history-attention-independent-20261127"
EVAL_CPU = SUPPORT.parent / "letter-beauty-history-attention-eval-cpu"


def put(source, relative):
    target = EVIDENCE / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def capture_tree(local, destination, suffixes):
    for source in sorted(local.rglob("*")):
        if (source.is_file() and not source.is_symlink() and source.suffix in suffixes
                and "__pycache__" not in source.parts):
            put(source, destination / source.relative_to(local))


for source in sorted((SUPPORT / "collected").iterdir()):
    folder = "runner" if source.name.startswith("letter-beauty-v2-history-attn-") else "stage"
    capture_tree(source, Path(folder, source.name), {".json", ".jsonl", ".log", ".txt"})
capture_tree(INDEPENDENT, Path("nonauthor"), {".json", ".log", ".py", ".sh", ".md"})
for name in ("check_inputs.py", "check_peft.py", "axis-parity.stdout", "axis-parity.stderr",
             "peft.stdout", "peft.stderr"):
    put(EVAL_CPU / name, Path("eval-cpu", name))
for name in ("submit.sh", "query.sh", "deploy.sh", "collect.sh", "preflight-sequence.sh",
             "preflight-completion.sh", "summarize.py", "stage-evidence.py", "readback.py"):
    put(SUPPORT / name, Path("scripts", name))
for name in ("deploy-b9dfa8f8adad6ee89c15bda9d978c5465ac7e772.log",
             "deploy-a26a9e9698fb5c66f04e93dd062f8beef8bade8b.log",
             "deploy-3ce7c70264454d16f4e9ffc8174ffac3b5a888e0.log", "live-resources-resume.txt", "cpu-r0.log", "round0-legacy-left-truncation.diff"):
    put(SUPPORT / "evidence" / name, Path("environment", name))
for commit in ("b9dfa8f8adad6ee89c15bda9d978c5465ac7e772", "a26a9e9698fb5c66f04e93dd062f8beef8bade8b",
               "3ce7c70264454d16f4e9ffc8174ffac3b5a888e0"):
    put(SUPPORT / f"export-{commit}" / "SOURCE.sha256", Path("environment", f"{commit}.SOURCE.sha256"))
put(SUPPORT / "summary-current.json", Path("summary.json"))
