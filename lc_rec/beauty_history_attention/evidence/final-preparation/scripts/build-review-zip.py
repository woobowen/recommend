"""Produce the final-preparation audit ZIP with explicit runtime dependency coverage."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import zipfile

HOME = Path("/home/addaswsw")
REPO = HOME / "recommend-worktrees/shuffle-v2"
SUPPORT = HOME / "recommend-support/beauty-history-attention-final-20261127"
INDEPENDENT = SUPPORT.parent / "letter-beauty-history-attention-formal-prep-independent-20261127"
PREVIOUS_ZIP = HOME / "recommend/LETTER_Beauty_LLaMA_history_attention_preflight_20261127.zip"
PREVIOUS_ZIP_SHA = "382996d0a0d7d55089840e21ead73fa03dc42c8de301dd97494c3acc5b8f1e9a"
PREVIOUS = "cac82870fd8374833f7df8cd0264543c245ede23"
BASELINE = "6e1b211d9c37f358c43b1540b40b17f6703150d2"
TASK = "lc_rec/beauty_history_attention"
A0 = "lc_rec/a0_beauty"
VENDOR = "lc_rec/letter/vendor/LETTER/LETTER-LC-Rec"
DEPS = [
    *[f"{A0}/{name}.py" for name in ("checkpoint_receipt", "g3_identity", "single_gpu_training",
                                      "trainer_loop_patch", "training_rows", "zero2_partial_state")],
    *[f"{VENDOR}/{name}" for name in ("modeling_letter.py", "data.py", "collator.py", "prompt.py",
                                       "utils.py", "evaluate.py", "test_ddp.py", "config/ds_z2_bf16.json")],
    *[f"lc_rec/letter/scripts/{name}.py" for name in ("compat", "formal_bnb_reload", "formal_evaluate",
                                                       "formal_attention")],
    "lc_rec/letter/evidence/formal_20261111/precision_approval.json",
    "lc_rec/letter/review/final_audit_20261113/statistics_verify.py",
    "lc_rec/letter/review/final_audit_20261113/external/official_cce.py",
    "lc_rec/shuffle/core.py",
    *[f"lc_rec/shuffle_v2/{name}.py" for name in ("beauty", "beauty_evaluate", "core", "evaluate",
                                                "reference", "replay", "runtime", "statistics")],
    "lc_rec/shuffle_v2/a0_beauty_letter.json",
]
HISTORICAL_RUNTIME = (
    "uneven_sampler.py", "preflight_users.py", "config/preflight_users_seed42.json",
    "normal_eval/a0/evaluation_config.json", "normal/a0/test/rank0.json",
    "runner-beauty-cbb963fab68c34cc.py", "runner-beauty-cbb963fab68c34cc.py.deployment.json",
)
REUSE = (
    "historical/server-originals/audit.json", "historical/server-originals/inputs.jsonl",
    "historical/server-originals/single-shard.json", "historical/server-originals/assets.json",
    "historical/server-originals/Beauty.index.json", "historical/server-originals/Beauty.inter.json",
    "historical/server-originals/Beauty.item.json",
    *[f"historical/server-originals/epoch{i}_rows.json" for i in range(1, 5)],
    "historical/actual-V2/metrics.csv", "historical/actual-V2/paired.csv",
    "historical/actual-V2/summary.json", "historical/actual-V2/formal-run.json",
    "historical/actual-A/events.rank0.jsonl", "historical/actual-A/events.rank1.jsonl",
    "historical/actual-A/best_selection.json",
    "eval-cpu/axis-parity.stdout", "eval-cpu/check_inputs.py", "actual-run/cpu-r1/cpu.json",
    "actual-run/train-r1/train-argv.json", "actual-run/train-resume-r1/train-argv.json",
    "actual-run/train-r1/training/rank0.json", "actual-run/train-r1/training/events.rank0.jsonl",
    "actual-run/train-r1/training/best_selection.json",
    *[f"actual-run/train-r1/training/checkpoint-{step}/complete.json" for step in (2, 7, 10)],
    "actual-run/letter-beauty-v2-history-attn-train-r1/status.json",
    "actual-run/letter-beauty-v2-history-attn-train-resume-r1/status.json",
    "actual-run/letter-beauty-v2-history-attn-train-resume-r1/request.json",
    "actual-run/letter-beauty-v2-history-attn-train-resume-r1/work.log",
    "actual-run/training-regression-r2/training-regression.json",
    "independent-review/FINAL.md",
    *[f"installed-compatibility-source/{name}" for name in (
        "transformers/models/llama/modeling_llama.py", "transformers/generation/utils.py",
        "transformers/trainer.py", "accelerate/data_loader.py", "accelerate/accelerator.py",
        "deepspeed/runtime/engine.py", "peft/peft_model.py", "peft/tuners/lora.py",
        "bitsandbytes/autograd/_functions.py", "fastchat/train/llama2_flash_attn_monkey_patch.py",
        "flash_attn/flash_attn_interface.py")],
)


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tree(commit, *paths):
    return git("ls-tree", "-r", "--name-only", commit, *paths).decode().splitlines()


def code_tree(commit):
    return [path for path in tree(commit, TASK) if "/evidence/" not in path]


class Pack:
    def __init__(self, archive):
        self.archive = archive
        self.files = {}
        self.seen = {}

    def add(self, name, content, origin):
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or str(path) != name or "\\" in name
                or ":" in name or name in self.files):
            raise ValueError(f"bad or duplicate archive path: {name}")
        sha = digest(content)
        row = {"sha256": sha, "bytes": len(content), "origin": origin}
        if sha not in self.seen:
            self.archive.writestr(name, content)
            self.seen[sha] = name
        else:
            row["duplicate_of"] = self.seen[sha]
        self.files[name] = row

    def local(self, name, path):
        self.add(name, path.read_bytes(), str(path))

    def repo(self, name, commit, path):
        self.add(name, git("show", f"{commit}:{path}"), f"{commit}:{path}")

    def local_tree(self, prefix, folder):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
                self.local(f"{prefix}/{path.relative_to(folder)}", path)


def build(path, args):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        pack = Pack(archive)
        pack.local("INDEX.md", SUPPORT / "INDEX.md")
        pack.local("REQUIREMENTS.md", SUPPORT / "requirements-original.md")
        pack.repo("README.md", args.release, f"{TASK}/README.md")
        for name in code_tree(args.release):
            if not name.endswith("/README.md"):
                pack.repo(f"published/{name}", args.release, name)
        for name in DEPS:
            pack.repo(f"runtime/{name}", args.snapshot, name)
        for name in HISTORICAL_RUNTIME:
            path_in_pack = (f"runtime-frozen/{A0}/{name}"
                            if name.split("/")[0] not in ("normal", "normal_eval") and not name.startswith("runner")
                            else f"runtime-frozen/{name}")
            pack.local(path_in_pack, SUPPORT / "frozen-runtime" / name)
        for name in ("formal_evaluate.py", "offline_metrics.py", "formal_train.py",
                     "config/formal_beauty_seed42.json", "epoch_schedule.py"):
            pack.repo(f"runtime-frozen/{A0}/{name}", BASELINE, f"{A0}/{name}")
        pack.repo(f"runtime-frozen/{TASK}/attention.py", PREVIOUS, f"{TASK}/attention.py")
        pack.repo("inherited/previous-experiment-README.md", PREVIOUS, f"{TASK}/README.md")
        pack.repo("inherited/previous-technical-summary.json", PREVIOUS, f"{TASK}/evidence/summary.json")
        assert digest(PREVIOUS_ZIP.read_bytes()) == PREVIOUS_ZIP_SHA
        with zipfile.ZipFile(PREVIOUS_ZIP) as old:
            previous_files = json.loads(old.read("MEMBERS.json"))["files"]
            for name in REUSE:
                row = previous_files[name]
                payload = old.read(row.get("duplicate_of", name))
                assert digest(payload) == row["sha256"]
                pack.add(f"inherited/{name}", payload, f"{PREVIOUS_ZIP.name}:{name}")
        pack.local_tree("actual", SUPPORT / "collected")
        pack.local_tree("transport", SUPPORT / "evidence")
        pack.local_tree("independent", INDEPENDENT)
        for name in ("deploy.sh", "submit.sh", "collect.sh", "query.sh", "pilot-sequence.sh", "readback.py",
                     "resource-estimate.py", "stage-evidence.py", "build-review-zip.py"):
            path_or_missing = SUPPORT / name
            if path_or_missing.is_file():
                pack.local(f"executed-tools/{name}", path_or_missing)
        for commit in dict.fromkeys([*args.run, args.snapshot]):
            pack.local(f"deployment/{commit}/SOURCE.sha256", SUPPORT / f"export-{commit}" / "SOURCE.sha256")
            for name in code_tree(commit):
                pack.repo(f"deployment/{commit}/{name}", commit, name)
        source_diff = git("diff", PREVIOUS, args.release, "--", TASK, f":(exclude){TASK}/evidence/**")
        pack.add("diffs/previous-to-final.patch", source_diff, f"git diff {PREVIOUS} {args.release}")
        final_diff = git("diff", args.snapshot, args.release, "--", TASK, f":(exclude){TASK}/evidence/**")
        pack.add("diffs/deployed-to-published.patch", final_diff, f"git diff {args.snapshot} {args.release}")
        archive.writestr("MEMBERS.json", json.dumps({
            "schema": 1, "published_commit": args.release, "formal_snapshot": args.snapshot,
            "preflight_snapshots": args.run, "files": pack.files,
        }, ensure_ascii=False, indent=2) + "\n")


def validate(path):
    patterns = re.compile(rb"(-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----|gh[opusr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})")
    with zipfile.ZipFile(path) as package:
        assert package.testzip() is None
        records = package.infolist()
        assert len(records) == len({row.filename.casefold() for row in records})
        entries = json.loads(package.read("MEMBERS.json"))["files"]
        for name, value in entries.items():
            got = package.read(value.get("duplicate_of", name))
            assert digest(got) == value["sha256"] and len(got) == value["bytes"], name
        for row in records:
            member = PurePosixPath(row.filename)
            assert not member.is_absolute() and ".." not in member.parts and "\\" not in row.filename
            assert not stat.S_ISLNK(row.external_attr >> 16)
            if member.suffix in (".py", ".sh", ".md", ".txt", ".json", ".jsonl", ".log", ".sha256"):
                assert not patterns.search(package.read(row)), row.filename
        required = [
            "INDEX.md", "REQUIREMENTS.md", "README.md",
            f"published/{TASK}/formal.py", f"published/{TASK}/train.py", f"published/{TASK}/evaluate.py",
            f"published/{TASK}/inference_attention.py", f"published/{TASK}/prediction_store.py",
            "runtime/lc_rec/shuffle_v2/evaluate.py",
            "runtime-frozen/lc_rec/a0_beauty/uneven_sampler.py",
            "runtime-frozen/lc_rec/a0_beauty/preflight_users.py",
            "runtime-frozen/lc_rec/a0_beauty/config/preflight_users_seed42.json",
            "runtime-frozen/normal/a0/test/rank0.json",
            "runtime-frozen/normal_eval/a0/evaluation_config.json",
            "actual/old-causal-r0/result.json", "actual/b-mask-r0/result.json",
            "actual/pilot/final/completion.json", "actual/metrics-r0/metrics/bounded-paired.json",
            "transport/github-readback.json", "transport/formal/b-formal-plan.unsigned.json",
            "transport/formal/gate-results.txt", "independent/FINAL.md",
            "transport/formal-final/b-formal-plan.unsigned.json", "transport/formal-final/gate-results.txt",
            "independent/ADDENDUM.md", "actual/letter-beauty-v2-history-attn-final-metrics-final-r0/work.log",
            "actual/pilot/chunks/000000/chunk.json",
            "actual/letter-beauty-v2-history-attn-final-generate-resume-duplicate-r0/work.log",
            "inherited/eval-cpu/axis-parity.stdout", "inherited/eval-cpu/check_inputs.py",
            "inherited/actual-run/cpu-r1/cpu.json", "inherited/actual-run/train-r1/train-argv.json",
            "inherited/actual-run/train-resume-r1/train-argv.json",
            "inherited/actual-run/letter-beauty-v2-history-attn-train-resume-r1/work.log",
            *[f"actual/pilot/final/{mode}/predictions.rank0.jsonl"
              for mode in ("normal", "shuffle2", "shuffle3a", "shuffle3b", "shuffle4")],
        ]
        assert all(name in entries for name in required)
        return {"members": len(records), "content_entries": len(entries),
                "known_required": len(required), "bytes": path.stat().st_size, "sha256": digest(path.read_bytes()),
                "crc_and_content": "PASS", "paths": "PASS", "credentials_scan": "PASS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    partial = args.output.with_suffix(".zip.partial")
    with partial.open("xb"):
        pass
    build(partial, args)
    stats = validate(partial)
    partial.rename(args.output)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
