"""Independent verifier for launch materialization and the first six G3 S2 rows."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import math
import re

GPU = "86660be5a62c8325caf74cf2810581850ca12803"
CPU = "cb3b1f36d69ef0c1a7f554ef5455026523635f78"
TRAINING = "21d495b0be5b606853ec7cec68a3eaa1b58acbd7"
MODEL = "3315622866cd97833741608ca21d82b72a10184ab8bff4dc688892146d0bf915"
CPU_MANIFEST = "2329944065d81f07c02041dbf5a62daf2609af7795e71804ed854db8faf58906"
FULL_ASSIGNMENT = "22ee58fe9f948f50e5cbc92160b6dc7df7e2f633cb9e8bf3121da5fb35bcb088"
CARD = "GPU-59cac3bd-3645-668b-79c8-839e0d878893"
PROTOCOL = "GRID-LETTER-SHUFFLE-V2"
PLACEHOLDER = "PIN_USER_AUTHORIZATION_SHA256_AFTER_WEB_REVIEW"


def obj(path):
    return json.loads(Path(path).read_bytes())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fsha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canon(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def runner_digest(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode())


def launch(args):
    p = obj(args.plan)
    a = obj(args.auth)
    parent = obj(args.parent)
    child = obj(args.child)
    authorization = fsha(args.auth)
    plan_sha = fsha(args.plan)

    assert p["formal_submitted"] is False and p["formal_gpu_device_seconds_delta"] == 0
    assert a["plan_sha256"] == plan_sha and a["web_review_pass"] is True
    assert a["scope"] == "G3_BEAUTY_SHUFFLE_V2_FORMAL" and a["authorized_by"] == "user"
    assert p["gpu_commit"] == a["run_commit"] == GPU
    assert p["evaluation_and_controller_commit"] == CPU and p["training_commit"] == TRAINING
    assert p["prepared_model_identity"] == MODEL
    assert p["planned_users_per_condition"] == 22363
    assert p["planned_candidates_per_condition"] == 447260

    expected = copy.deepcopy(p["parent"]["request"])
    expected["argv"][expected["argv"].index(PLACEHOLDER)] = authorization
    expected["inputs"] = p["release_inputs"][CPU] + expected["inputs"] + [
        {"path": p["server_plan_path"], "sha256": plan_sha},
        {"path": p["approval_path"], "sha256": authorization},
    ]
    # The materialization order of the two separately pinned files is immaterial.
    assert all(parent[key] == value for key, value in expected.items() if key != "inputs")
    assert len(parent["inputs"]) == len(expected["inputs"])
    assert {tuple(sorted(x.items())) for x in parent["inputs"]} == {
        tuple(sorted(x.items())) for x in expected["inputs"]
    }

    assert tuple(p["modes"]) == ("shuffle2", "shuffle3a", "shuffle3b", "shuffle4")
    expected_gpu = p["modes"]["shuffle2"]["gpu"]
    assert expected_gpu["predecessor_members"] == []
    expected = json.loads(json.dumps(expected_gpu["request"]).replace(PLACEHOLDER, authorization))
    expected["inputs"] = p["release_inputs"][GPU] + expected["inputs"]
    assert child == expected
    assert parent["env"]["CUDA_VISIBLE_DEVICES"] == "" and parent["env"]["DS_ACCELERATOR"] == "cpu"
    assert child["env"]["CUDA_VISIBLE_DEVICES"] == p["card"] and child["env"]["DS_ACCELERATOR"] == "cuda"
    assert all(phase["gpu"]["request"]["env"]["CUDA_VISIBLE_DEVICES"] == p["card"]
               and phase["cpu"]["request"]["env"]["CUDA_VISIBLE_DEVICES"] == ""
               and phase["cpu"]["request"]["env"]["DS_ACCELERATOR"] == "cpu"
               for phase in p["modes"].values())
    assert "--nproc_per_node=1" in child["argv"]
    assert child["argv"][child["argv"].index("--scope") + 1] == "FORMAL"
    assert not any("--preflight" in value for value in child["argv"])

    guard_reservation = sum(phase["gpu"]["request"]["timeout_seconds"] + 15
                            for phase in p["modes"].values())
    cpu_reservation = sum(phase["cpu"]["request"]["timeout_seconds"]
                          for phase in p["modes"].values()) + p["paired"]["request"]["timeout_seconds"]
    assert guard_reservation == 115200 and cpu_reservation == 7200
    assert p["observed_prior_formal_device_seconds"] + guard_reservation <= 129600
    assert guard_reservation + cpu_reservation < parent["timeout_seconds"] == 126000
    return {"status": "PASS", "scope": "launch immutable request materialization",
            "plan_sha256": plan_sha, "authorization_sha256": authorization,
            "parent_request_canonical_sha256": runner_digest(parent),
            "child_request_canonical_sha256": runner_digest(child),
            "gpu_guard_reservation_seconds": guard_reservation,
            "cpu_request_reservation_seconds": cpu_reservation,
            "parent_timeout_seconds": parent["timeout_seconds"], "gpu_uuid": p["card"]}


def first_rows(args):
    rank = obj(args.rank)
    binding = obj(args.binding)
    audit = obj(args.cpu_audit)
    index = obj(args.index)
    inter = obj(args.inter)
    raw_lines = Path(args.samples).read_bytes().splitlines(keepends=True)
    assert len(raw_lines) == 6 and all(line.endswith(b"\n") for line in raw_lines)
    rows = list(map(json.loads, raw_lines))
    manifest_sha = fsha(args.cpu_manifest)
    assert manifest_sha == CPU_MANIFEST == audit["manifest_sha256"]
    assert all(fsha(path) == audit["assets"][kind]["sha256"]
               for kind, path in (("index", args.index), ("inter", args.inter)))
    assert audit["status"] == "PASS_G3_CPU_V2_FULL" and audit["users"] == 22363
    assert (audit["items"], audit["sids"], audit["collisions"]) == (12101, 12101, 0)
    axis = rank["assigned_row_indices"]
    assert (rank["scope"], rank["assigned_users"], rank["preflight_only"], rank["condition"]) == ("FORMAL", 22363, False, "shuffle2")
    assert sorted(axis) == list(range(22363))
    assert rank["assignment_sha256"] == audit["assignment_sha256"] == FULL_ASSIGNMENT == sha(canon(axis))
    assert [row["row_idx"] for row in rows] == axis[:6]
    assert rank["RUN_COMMIT"] == binding["V2_RUN_COMMIT"] == GPU
    assert rank["MODEL_TRAINING_COMMIT"] == binding["MODEL_TRAINING_COMMIT"] == TRAINING
    assert rank["rank"] == rank["local_rank"] == 0 and rank["world_size"] == 1
    assert rank["model_identity_sha256"] == MODEL
    assert rank["tokenizer_identity"] == audit["tokenizer_identity"]
    assert rank["adapter_identity"]["scope"] == "FORMAL" and rank["adapter_identity"]["run_commit"] == TRAINING
    original_callback = "19b381838a03cc2273e6e2baf75606520ffe77f28e2525bc791578542d5bd93e"
    assert rank["allowed_callback_sha256"] == original_callback
    generation = rank["generation"]
    assert (generation["prompt_id"], generation["per_rank_batch_size"], generation["num_beams"],
            generation["num_return_sequences"], generation["max_new_tokens"], generation["early_stopping"],
            generation["filter_items"]) == (0, 1, 20, 20, 10, True, False)
    assert binding["cpu_audit_sha256"] == fsha(args.cpu_audit)
    assert binding["cpu_manifest_sha256"] == manifest_sha
    assert binding["scope"] == binding["adapter_scope"] == "FORMAL"
    assert binding["mode"] == "shuffle2" and not any("--preflight" in part for part in binding["effective_argv"])
    assert binding["derived_sha256"] == fsha(args.derived)
    adapter = rank["strict_adapter_reload"]
    assert (adapter["keys"], adapter["lora_keys"], adapter["elements"]) == (450, 448, 288571392)
    model = rank["v2_model"]
    assert (model["base_dtype"], model["adapter_embedding_and_head_dtype"], model["lora_dtype"]) == (
        "torch.bfloat16", "torch.bfloat16", "torch.float32"
    )
    assert model["all_parameters_on_logical_cuda_0"] and model["eval_mode"] and model["requires_grad_false"]
    for key, value in {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0",
                       "CUDA_VISIBLE_DEVICES": args.card, "DS_ACCELERATOR": "cuda"}.items():
        assert model["rank_environment"][key] == value

    tokens = audit["token_ids"]
    reverse = {token: symbol for symbol, token in tokens.items()}
    catalog = {"".join(sid) for sid in index.values()}
    assert len(tokens) == len(reverse) == 786 and len(catalog) == 12101
    allowed = [{tokens[sid[i]] for sid in index.values()} for i in range(4)]
    ids = set(axis[:6])
    with Path(args.cpu_manifest).open() as stream:
        cpu_rows = {n: json.loads(line) for n, line in enumerate(stream) if n in ids}
    findings = []
    for row in rows:
        r = row["row_idx"]
        expected = cpu_rows[r]
        original, moved, mask = (row[k] for k in ("original_input_token_ids", "input_token_ids", "attention_mask"))
        sid_sequence = inter[row["user_id"]]
        target = str(sid_sequence[-1])
        history = list(map(str, sid_sequence[:-1][-20:]))
        assert target == row["target_item_id"] and row["target"] == "".join(index[target])
        assert all(row[k] == expected[k] for k in ("user_id", "target_item_id", "target", "row_idx"))
        assert row["scope"] == "FORMAL" and row["condition"] == "shuffle2" and row["V2_RUN_COMMIT"] == GPU
        assert row["rank"] == 0 and row["world_size"] == 1 and row["prompt_id"] == 0
        assert row["model_identity_sha256"] == MODEL and row["preflight_extra_left_padding"] == 0
        assert mask == [1] * len(original) and len(moved) == len(original) == row["input_nonpadding_tokens"]
        assert row["lookup_equivalence_exact"] and row["original_callback_verified_all_tokens"]
        key = row["v2_cpu_binding"]
        assert key == {"manifest_sha256": manifest_sha, "row_sha256": sha(canon(expected)),
                       "original_sha256": sha(canon(original)), "input_sha256": sha(canon(moved)),
                       "attention_mask_sha256": sha(canon(mask)), "mode": "shuffle2", "row_idx": r}
        assert key["original_sha256"] == expected["original_sha256"]
        assert key["attention_mask_sha256"] == expected["attention_mask_sha256"]
        mode = expected["conditions"]["shuffle2"]
        assert key["input_sha256"] == mode["input_sha256"] and mode["input_changed"] == (original != moved)

        pre = row["actual_model_prefill"]
        assert pre["input_ids"] == moved and pre["attention_mask"] == mask
        pos = list(range(len(moved)))
        assert pre["position_ids"] == pos and pre["initial_cache_length"] == 0
        assert pre["beam_expansion"] == 20 and pre["effective_use_cache"] is True
        for name, value in [("input", moved), ("mask", mask), ("position", pos)]:
            assert pre[f"all_beam_{name}_sha256"] == sha(canon([value] * 20))
        lookup = pre["embedding_lookup"]
        assert lookup["shape"] == [20, len(moved), 4096] and lookup["all_beams_exact"] is True
        assert lookup["dtype"] == "torch.bfloat16" and lookup["input_sha256"] == key["input_sha256"]
        assert re.fullmatch(r"[0-9a-f]{64}", lookup["first_beam_bf16_sha256"])

        shuffle = row["shuffle"]
        assert (shuffle["protocol"], shuffle["identity_policy"], shuffle["seed"],
                shuffle["mode"], shuffle["active_positions"], shuffle["row_idx"]) == (
            PROTOCOL, "allow_identity", 42, "shuffle2", [2, 3], r
        )
        full = original.copy()
        groups = shuffle["evidence"]["groups"]
        assert len(history) == len(groups) == len(expected["groups"])
        for j, (item, group, frozen) in enumerate(zip(history, groups, expected["groups"], strict=True)):
            old_ids = [tokens[symbol] for symbol in index[item]]
            positions = group["positions"]
            assert [item, positions, old_ids] == frozen
            assert positions == list(range(positions[0], positions[0] + 4))
            assert old_ids == [original[i] for i in positions] == group["original_ids"]
            # For two slots, 2**256 is exactly divisible by 2. No draw is rejected.
            pick = int.from_bytes(hashlib.sha256(canon([PROTOCOL, 42, 2, r, j, 0])).digest(), "big") % 2
            assert group["visible_history_item_index"] == j and group["rejection_counter"] == 0
            assert group["permutation_id"] == pick == mode["permutation_ids"][j]
            perm = [0, 1, 3, 2] if pick else [0, 1, 2, 3]
            assert group["permutation"] == perm and group["relative_permutation"] == ([1, 0] if pick else [0, 1])
            new_ids = [old_ids[i] for i in perm]
            assert group["shuffled_ids"] == new_ids
            for offset, token in zip(positions, new_ids, strict=True):
                full[offset] = token
        assert full == moved

        effective = row["effective_generation"]
        assert effective["native_score_dtype"] == "torch.float32"
        kwargs = {"max_new_tokens": 10, "num_beams": 20, "num_return_sequences": 20,
                  "output_scores": True, "return_dict_in_generate": True, "early_stopping": True}
        defaults = {"use_cache": True, "do_sample": False, "eos_token_id": 2, "pad_token_id": 0}
        assert effective["kwargs"] == kwargs and effective["effective"] == kwargs | defaults
        assert all(effective["generation_config"][k] == v for k, v in defaults.items())
        cands = row["candidates"]
        assert len(cands) == 20
        for j, candidate in enumerate(cands):
            suffix = candidate["generated_token_ids"]
            assert (candidate["generation_order"], candidate["eos_position"]) == (j, 4)
            assert len(suffix) >= 5 and len(suffix) <= 10 and suffix[4] == 2
            assert all(tok in allowed[pos] for pos, tok in enumerate(suffix[:4]))
            assert set(suffix[5:]) <= {0, 2}
            sid = "".join(reverse[tok] for tok in suffix[:4])
            assert candidate["sid"] == sid == candidate["raw_decoded"].split("Response:")[-1].strip().replace(" ", "")
            assert math.isfinite(candidate["sequence_score"])
        order = sorted(range(20), key=lambda i: cands[i]["sequence_score"], reverse=True)
        assert row["sorted_generation_order"] == order
        assert row["relevance"] == [int(cands[i]["sid"] == row["target"]) for i in order]
        findings.append({"row_idx": r, "candidates": 20, "history_groups": len(groups),
                         "catalog_invalid_retained": sum(candidate["sid"] not in catalog for candidate in cands),
                         "cpu_manifest_row_sha256": key["row_sha256"], "bytes_sha256": sha(raw_lines[len(findings)])})
    return {"status": "PASS", "scope": "first six complete formal S2 records only",
            "first_six_sha256": fsha(args.samples), "first_six_bytes": Path(args.samples).stat().st_size,
            "cpu_manifest_sha256": manifest_sha, "rank_report_sha256": fsha(args.rank),
            "binding_sha256": fsha(args.binding), "derived_sha256": fsha(args.derived),
            "users": findings, "total_candidates": 120, "model_independent_forward": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    material = commands.add_parser("launch")
    for key in ("plan", "auth", "parent", "child"):
        material.add_argument("--" + key, required=True)
    rows = commands.add_parser("first-rows")
    for key in ("rank", "binding", "cpu-audit", "cpu-manifest", "index", "inter", "samples", "derived"):
        rows.add_argument("--" + key, required=True)
    rows.add_argument("--card", default=CARD)
    args = parser.parse_args()
    print(json.dumps(launch(args) if args.command == "launch" else first_rows(args), indent=2))


if __name__ == "__main__":
    main()
