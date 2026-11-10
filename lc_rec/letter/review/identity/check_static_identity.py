#!/usr/bin/env python3
"""Bounded CPU-only provenance and protocol checks; never imports model code."""
import ast
import collections
import hashlib
import json
import math
import pathlib
import re
import shlex

ROOT = pathlib.Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor/LETTER"
REVIEW = ROOT / "review/identity"


def main():
    tree = json.loads((REVIEW / "fixed_source_tree.json").read_text())
    expected = {e["path"]: e["sha"] for e in tree["tree"] if e["type"] == "blob"}
    source_files = []
    for path in sorted(VENDOR.rglob("*")):
        if not path.is_file():
            continue
        content = path.read_bytes()
        relative = str(path.relative_to(VENDOR))
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        source_files.append({"path": relative, "bytes": len(content), "git_blob": blob,
                             "expected_git_blob": expected.get(relative), "match": blob == expected.get(relative)})
    defaults = {}
    utils = ast.parse((VENDOR / "LETTER-LC-Rec/utils.py").read_text())
    for node in ast.walk(utils):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        flags = [ast.literal_eval(a) for a in node.args if isinstance(a, ast.Constant)]
        for keyword in node.keywords:
            if keyword.arg == "default":
                defaults[flags[0].lstrip("-")] = ast.literal_eval(keyword.value)
    script = (VENDOR / "LETTER-LC-Rec/run_train.sh").read_text().replace("\\\n", " ")
    assignments = {}
    for line in script.splitlines():
        match = re.match(r"^([A-Z_]+)=(.*)$", line)
        if match:
            tokens = shlex.split(match.group(2), comments=True)
            value = tokens[0] if tokens else ""
            assignments[match.group(1)] = re.sub(r"\$([A-Z_]+)", lambda m: assignments.get(m[1], m[0]), value)
    command = shlex.split(script[script.index("torchrun"):])
    cli = {}
    for i, arg in enumerate(command):
        if arg.startswith("--"):
            value = command[i + 1] if i + 1 < len(command) and not command[i + 1].startswith("--") else True
            cli[arg[2:]] = re.sub(r"\$([A-Z_]+)", lambda m: assignments.get(m[1], m[0]), value) if isinstance(value, str) else value
    effective = {**defaults, **cli}
    document = ROOT / "evidence/author_hyperparameters.txt"
    author = [ast.literal_eval(s) for s in document.read_text(encoding="utf-8-sig").splitlines()
              if s.startswith("{")][0]
    comparison = {}
    for key, value in author.items():
        public_value = effective.get(key)
        if isinstance(value, (float, int)) and not isinstance(value, bool) and public_value is not None:
            public_value = type(value)(public_value)
        comparison[key] = {"author_record": value, "public_cli_or_default": public_value, "different": value != public_value}
    prompts = {}
    exec(compile((VENDOR / "LETTER-LC-Rec/prompt.py").read_text(), "upstream_prompt.py", "exec"), prompts)
    index = json.loads((VENDOR / "data/Instruments/Instruments.index.json").read_text())
    alphabets = [sorted({sid[i] for sid in index.values()}) for i in range(4)]
    sid_to_items = collections.defaultdict(list)
    for item, sid in index.items():
        sid_to_items[tuple(sid)].append(item)
    values = set(sid_to_items)
    witness = None
    for a in alphabets[0]:
        for b in alphabets[1]:
            for c in alphabets[2]:
                for d in alphabets[3]:
                    if (a, b, c, d) not in values:
                        witness = [a, b, c, d]
                        break
                if witness:
                    break
            if witness:
                break
        if witness:
            break
    collisions = [{"sid": list(sid), "items": items}
                  for sid, items in sid_to_items.items() if len(items) > 1]
    inters = json.loads((VENDOR / "data/Instruments/Instruments.inter.json").read_text())
    syntax = []
    for path in sorted((VENDOR / "LETTER-LC-Rec").glob("*.py")):
        ast.parse(path.read_text())
        syntax.append(str(path.relative_to(VENDOR)))
    result = {
        "kind": "CPU static identity verification, no model load or metrics",
        "upstream_commit": tree["sha"],
        "all_materialized_files_match_fixed_tree": all(x["match"] for x in source_files),
        "source_files": source_files, "syntax_parsed_files": syntax,
        "doc_sha256": hashlib.sha256(document.read_bytes()).hexdigest(),
        "author_instruments": author, "public_train_argv_lexed": command,
        "public_shell_assignments": assignments, "public_base_model_placeholder_empty": not assignments["BASE_MODEL"],
        "author_vs_public": comparison, "seqrec_prompt_count": len(prompts["all_prompt"]["seqrec"]),
        "index_items": len(index), "index_unique_sid": len(values),
        "index_collision_groups": len(collisions),
        "index_items_in_collision_groups": sum(len(x["items"]) for x in collisions),
        "index_extra_collisions": sum(len(x["items"]) - 1 for x in collisions),
        "index_collision_examples": collisions[:4],
        "level_token_counts": [len(a) for a in alphabets],
        "positionwise_candidate_count": math.prod(len(a) for a in alphabets),
        "positionwise_valid_candidate_ratio": len(values) / math.prod(len(a) for a in alphabets),
        "positionwise_invalid_sid_witness": witness,
        "witness_tokens_all_allowed_at_respective_positions": all(witness[i] in alphabets[i] for i in range(4)),
        "witness_not_in_index": tuple(witness) not in values,
        "user_count": len(inters), "interaction_count": sum(map(len, inters.values())),
        "seqrec_training_records": sum(max(0, len(items) - 3) for items in inters.values()),
        "seqrec_validation_records_at_two_sampled_prompts": 2 * len(inters),
        "seqrec_test_records": len(inters),
    }
    assert result["all_materialized_files_match_fixed_tree"]
    assert result["witness_tokens_all_allowed_at_respective_positions"] and result["witness_not_in_index"]
    (REVIEW / "static_verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ["all_materialized_files_match_fixed_tree", "seqrec_prompt_count",
          "index_items", "index_unique_sid", "index_collision_groups", "index_extra_collisions",
          "positionwise_invalid_sid_witness", "seqrec_training_records", "seqrec_test_records"]}, indent=2))


if __name__ == "__main__":
    main()
