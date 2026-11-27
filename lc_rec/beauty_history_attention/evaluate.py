"""Resumable B-only LETTER Beauty generation and frozen Shuffle V2 evaluation."""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import datetime as dt
import difflib
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
import time
import traceback
from types import SimpleNamespace

from lc_rec.beauty_history_attention.identity import (
    IDENTITY_FILE, require, tokenizer_identity, verify_adapter_identity, verify_base_model,
)
from lc_rec.beauty_history_attention.prediction_store import (
    CHUNK_USERS, CONDITIONS, PREDICTIONS, SPEC, atomically_write_json, fsync_dir, open_store, verify_final,
)
from lc_rec.shuffle.core import transform as verify_history_groups
from lc_rec.shuffle_v2 import beauty, core, statistics
from lc_rec.shuffle_v2.beauty_evaluate import derive_metrics
from lc_rec.shuffle_v2.replay import frozen, semantic_hash

REPLAY_DIR = Path('/data/bwwu/lc-rec/letter-shuffle-v2-beauty/cpu-replay-r3')
# The three source hashes are from the actual historical V2 formal request.
REPLAY_SHA = {
    'audit.json': 'f5802e83be477126eccde54a1945ffacd900c9f4a9fa30999ac0e2543569a57a',
    'inputs.jsonl': '2e69ea72e47eb978e3d7375c47aa62cd3d3588fa633bdf7b6d1cf7000adc23b2',
    'single-shard.json': '00f4d70d297611a822191e7f3fa4aba45d4d657a2207e6a67824eba4b73cf2cd',
}
SOURCE_V2 = '1182dfd914c20bd3dc11786587eae6879fba6b9d'
CHECKPOINT = IDENTITY_FILE


def read(path):
    return json.loads(path.read_text())


def frozen_json(config, path):
    raw = path.read_bytes()
    frozen_identity = next(item["sha256"] for item in config["files"] if item["path"] == str(path))
    require(hashlib.sha256(raw).hexdigest() == frozen_identity, f"frozen JSON identity changed: {path}")
    return json.loads(raw)


def check_commit(value):
    if len(value) != 40 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Expected a complete lowercase commit SHA')


def read_replay(root, config):
    for name, digest in REPLAY_SHA.items():
        if frozen.sha(root / name) != digest:
            raise ValueError(f'Frozen Beauty V2 replay hash differs for {name}')
    audit, shard = (read(root / n) for n in ('audit.json', 'single-shard.json'))
    with (root / 'inputs.jsonl').open() as stream:
        rows = list(map(json.loads, stream))
    assert audit['status'] == 'PASS_CPU_V2_FULL' and audit['protocol'] == core.PROTOCOL
    assert shard['status'] == 'PASS_SINGLE_SHARD_CPU'
    assert len(rows) == audit['users'] == shard['users'] == config['users'] == 22363
    assert audit['user_axis_sha256'] == config['user_axis_sha256']
    assert audit['tokenizer_vocab_sha256'] == config['vocab_sha256']
    assert audit['manifest_sha256'] == shard['manifest_sha256'] == REPLAY_SHA['inputs.jsonl']
    assert all(row['row_idx'] == i for i, row in enumerate(rows))
    return audit, rows, shard


MAX_PREFLIGHT_USERS = 64


def preflight_rows(rows, budget=None):
    # Decide from lengths and intervention identity only, before seeing scores.
    selected = [min(rows, key=lambda row: row['encoded_length']),
                max(rows, key=lambda row: row['encoded_length'])]
    for mode in core.MODES:
        identity = next((row for row in rows if not row['conditions'][mode]['input_changed']), None)
        if identity is not None:
            selected.append(identity)
    boundary = list(dict.fromkeys(row["row_idx"] for row in selected))
    if budget is None:
        return boundary
    require(len(boundary) <= budget <= min(MAX_PREFLIGHT_USERS, len(rows)), "invalid bounded sampling budget")
    excluded = set(boundary)
    # Stratify by exact encoded length without examining scores or candidate output.
    axis = sorted((row for row in rows if row["row_idx"] not in excluded),
                  key=lambda row: (row["encoded_length"], row["row_idx"]))
    extra = budget - len(boundary)
    return boundary + [axis[((2 * i + 1) * len(axis)) // (2 * extra)]["row_idx"]
                       for i in range(extra)]


def paired_inputs(raw, mask, groups, sid_vocab, replay_row):
    index = replay_row['row_idx']
    canonical, verified = verify_history_groups(raw, mask, groups, sid_vocab, 0, index)
    assert canonical == raw and len(raw) == replay_row['encoded_length']
    assert verified['original_sha256'] == replay_row['original_sha256']
    assert verified['attention_mask_sha256'] == replay_row['attention_mask_sha256']
    assert len(groups) == replay_row['visible_history_length']
    positions = [position for group in verified['groups'] for position in group['positions']]
    history_slots = set(positions)
    history = [i in history_slots for i in range(len(raw))]
    assert sum(history) == 4 * len(groups) and all(mask[i] for i in positions)
    common = dict(row_idx=index, original_sha256=verified['original_sha256'],
                  attention_mask_sha256=verified['attention_mask_sha256'],
                  cpu_manifest_sha256=REPLAY_SHA['inputs.jsonl'], history_group_count=len(groups),
                  history_sid_mask_sha256=core.canonical_sha256(history))
    conditions = {'normal': (raw, {**common, 'mode': 'normal', 'input_changed': False,
                                  'shuffled_sha256': common['original_sha256'], 'permutation_ids': []})}
    for mode in core.MODES:
        ids, evidence = core.transform(raw, mask, groups, sid_vocab, mode, index)
        assert [position for item in evidence['groups'] for position in item['positions']] == positions
        expected = replay_row['conditions'][mode]
        perm_ids = [group['permutation_id'] for group in evidence['groups']]
        assert (evidence['shuffled_sha256'], evidence['input_changed'], perm_ids) == (
            expected['input_sha256'], expected['input_changed'], expected['permutation_ids'])
        conditions[mode] = (ids, {**common, **core.mode_metadata(mode), 'permutation_ids': perm_ids,
                                 'input_changed': evidence['input_changed'],
                                 'shuffled_sha256': evidence['shuffled_sha256']})
    return history, conditions


def checkpoint_binding(adapter, baseline, purpose, identities):
    checkpoint = verify_adapter_identity(adapter, scope=purpose, mode='bidirectional')
    assert checkpoint['model_revision'] == baseline['model_identity']['revision']
    assert checkpoint['data_identity'] == {kind: entry['sha256'] for kind, entry in identities.items()}
    assert checkpoint['adapter_sha256'] != baseline['adapter_identity']['sha256']
    return checkpoint


def generation_settings(model):
    params = dict(num_beams=20, num_return_sequences=20, max_new_tokens=10, early_stopping=True,
                  do_sample=False, output_scores=True, return_dict_in_generate=True)
    effective = deepcopy(model.generation_config)
    assert not effective.update(**params)
    assert effective.use_cache is True
    assert (effective.bos_token_id, effective.eos_token_id, effective.pad_token_id) == (1, 2, 0)
    return params, effective.to_dict()


def load_inference(adapter, report, config, source, author, index):
    """Strictly reload the frozen BF16/FP32 inference path before installing a mode."""
    import torch

    baseline = config['baseline']
    report['verified_base_files'] = verify_base_model(baseline['model'])
    started = time.monotonic()
    model_args = SimpleNamespace(adapter=str(adapter), model=baseline['model'])
    model, tokenizer = source.strict_reload(model_args, report, torch, 0, author, index)
    torch.cuda.synchronize()
    report['load_and_strict_verification_seconds'] = time.monotonic() - started
    assert report['model'] == baseline['model_identity']
    historical = frozen_json(config, Path(baseline['normal_test_dir']) / 'rank0.json')
    actual_checks = report['strict_adapter_reload']['tensor_checks']
    assert {key: value['loaded_dtype'] for key, value in actual_checks.items()} == {
        key: value['loaded_dtype'] for key, value in historical['strict_adapter_reload']['tensor_checks'].items()}
    model.requires_grad_(False)
    base = model.get_base_model()
    assert not getattr(base, 'is_loaded_in_4bit', False) and not getattr(base, 'is_loaded_in_8bit', False)
    assert not model.training and all(p.device == torch.device('cuda', 0) for p in model.parameters())
    assert base.model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
    return model, tokenizer


def predict(model, base, tokenizer, source, allowed, ids, mask, history, settings, capture):
    import torch
    positions = mask.long().cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 1)
    prompt_length = ids.shape[1]
    boundary, decodes = {}, []

    def before_forward(module, positional, named):
        seen = named['input_ids']
        seen_mask = named['attention_mask']
        position_ids = named['position_ids']
        past = named.get('past_key_values')
        seen_history = named['history_sid_mask']
        assert seen_history.dtype == torch.bool and seen_history.shape[0] == seen.shape[0] == 20
        assert torch.equal(seen_history[:, :prompt_length], history.expand(20, -1))
        if not boundary:
            assert past is None and torch.equal(seen, ids.expand(20, -1))
            assert torch.equal(seen_mask, mask.expand(20, -1))
            assert torch.equal(position_ids, positions.expand(20, -1))
            assert seen_history.shape == seen.shape
            boundary.update(input_sha256=core.canonical_sha256(seen[0].tolist()),
                            attention_mask_sha256=core.canonical_sha256(seen_mask[0].tolist()),
                            position_ids_sha256=core.canonical_sha256(position_ids[0].tolist()),
                            history_sid_mask_sha256=core.canonical_sha256(seen_history[0].tolist()),
                            cache_length=0, beam_expansion=20)
            if capture:
                boundary.update(input_ids=seen[0].tolist(), attention_mask=seen_mask[0].tolist(),
                                position_ids=position_ids[0].tolist(), history_sid_mask=seen_history[0].tolist())
            return
        cache_length = int(past[0][0].shape[-2])
        assert seen.shape[1] == 1 and seen_mask.shape[1] == cache_length + 1
        assert seen_history.shape[1] == prompt_length and cache_length >= prompt_length
        expected_pos = seen_mask.long().sum(dim=-1, keepdim=True) - 1
        assert torch.equal(position_ids, expected_pos)
        if capture:
            decodes.append({'cache_length': cache_length, 'position_ids': position_ids[:, 0].tolist(),
                            'history_length': seen_history.shape[1], 'query_length': seen.shape[1]})

    with ExitStack() as stack:
        if capture:
            embedding = model.get_input_embeddings()
            lookup = embedding(ids)

            def verify_lookup(module, positional, embedding_output):
                if 'lookup_exact' not in boundary:
                    assert boundary and torch.equal(embedding_output, lookup.expand_as(embedding_output))
                    boundary['lookup_exact'] = True

            stack.callback(embedding.register_forward_hook(verify_lookup).remove)
        stack.callback(base.register_forward_pre_hook(before_forward, with_kwargs=True).remove)
        tick = time.monotonic()
        generated = model.generate(input_ids=ids, attention_mask=mask, history_sid_mask=history,
                                   prefix_allowed_tokens_fn=allowed, **settings)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - tick
    assert boundary
    if capture:
        assert len(decodes) == 4 and boundary['lookup_exact']
        boundary['decodes'] = decodes
    input_ids = ids[0].tolist()
    predictions = generated.sequences.cpu().tolist()
    scores = generated.sequences_scores.float().cpu().tolist()
    decoded = tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)
    assert len(predictions) == len(scores) == len(decoded) == 20 and all(map(math.isfinite, scores))
    candidates = []
    for order, (sequence, score, text) in enumerate(zip(predictions, scores, decoded, strict=True)):
        assert sequence[:prompt_length] == input_ids
        suffix = sequence[prompt_length:]
        eos = suffix.index(tokenizer.eos_token_id)
        assert eos == 4
        for position, token in enumerate(suffix[:eos + 1]):
            permitted = allowed(0, torch.tensor(input_ids + suffix[:position], dtype=torch.long))
            assert permitted is not None and token in permitted
        assert all(token in {tokenizer.eos_token_id, tokenizer.pad_token_id} for token in suffix[eos + 1:])
        candidates.append(dict(generation_order=order, generated_token_ids=suffix, eos_position=eos,
                               raw_decoded=text, sequence_score=score, sid=source.normalized_prediction(text)))
    return candidates, sorted(range(20), key=scores.__getitem__, reverse=True), elapsed, boundary


def _formal_generation_guard(args):
    if args.purpose == "preflight":
        require(args.plan is None and args.formal_authorization is None
                and args.formal_authorization_sha256 is None, "formal gate arguments are only used for formal B")
        require(args.sample_budget is None or 1 <= args.sample_budget <= MAX_PREFLIGHT_USERS,
                "preflight sample size exceeds the bounded budget")
        require(args.stop_after_chunks is None or args.stop_after_chunks > 0, "invalid preflight stop trigger")
        return None
    require(args.plan is not None and args.formal_authorization is not None
            and args.formal_authorization_sha256, "formal B generation requires the reviewed plan and distinct user authorization")
    require(args.sample_budget is None and args.stop_after_chunks is None and args.chunk_users == CHUNK_USERS,
            "formal B uses only the complete canonical assignment and fixed checkpoint granularity")
    from lc_rec.beauty_history_attention.formal import verify_gate, verify_selected_adapter

    plan = verify_gate(args.plan, args.formal_authorization, args.formal_authorization_sha256)
    require(args.run_commit == plan["run_commit"] and args.backend == plan["backend"],
            "generation snapshot or backend differs from the authorized plan")
    require(args.config.resolve() == Path(plan["evaluation_config"]).resolve()
            and args.replay_dir.resolve() == Path(plan["replay_dir"]).resolve(),
            "generation configuration or replay path differs from the authorized plan")
    require(args.output.resolve() == Path(plan["output_root"]) / "generation",
            "formal generation output differs from the authorized plan")
    require(args.max_seconds == plan["budgets"]["generation_seconds"]
            and os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu_uuid"],
            "generation budget or GPU differs from the authorized plan")
    verify_selected_adapter(plan, args.adapter)
    return frozen.sha(args.plan)


def verify_completed_generation(root, run_commit, adapter, backend):
    """Formal phase reuse validates its merged files and the same selected B manifest."""
    report = verify_final(root)
    require(report["status"] == "PASS" and report["purpose"] == "formal", "formal generation is incomplete")
    require(report["RUN_COMMIT"] == run_commit and report["backend"] == backend
            and report["checkpoint_manifest_sha256"] == frozen.sha(Path(adapter) / CHECKPOINT),
            "generation completion differs from active model identity")
    return report


def _install_inference(base, backend, mode):
    if backend == "eager":
        from lc_rec.beauty_history_attention.inference_attention import install_eager_inference

        return install_eager_inference(base, mode)
    from lc_rec.beauty_history_attention.attention import install_attention

    return install_attention(base, mode)


def generate(args):
    require(not sys.flags.optimize, "Python optimization disables verified generation guards")
    started = time.monotonic()
    check_commit(args.run_commit)
    require(0 < args.max_seconds <= 96 * 3600, "invalid generation wall budget")
    require(1 <= args.chunk_users <= CHUNK_USERS, "invalid generation chunk user count")
    # No CUDA import, output directory, or model load before the formal gate.
    plan_sha = _formal_generation_guard(args)
    config = read(args.config)
    baseline = config["baseline"]
    output = args.output.absolute()
    source_file = Path(baseline["entry"])
    source = beauty.frozen_dependency(config, source_file)
    author = source.author_source(baseline["source-root"])
    for item in [*baseline["author_sources"].values(), next(
            item for item in config["files"] if item["path"] == str(author / "utils.py"))]:
        assert frozen.sha(Path(item["path"])) == item["sha256"]
    index, catalog, users, identities = source.catalog_and_users(baseline["data-root"])
    audit, manifest_rows, sampler = read_replay(args.replay_dir, config)
    assert semantic_hash([row["user_id"] for row in users]) == config["user_axis_sha256"]
    assert all((row["user_id"], row["target"]) == (other["user_id"], other["target"])
               for row, other in zip(users, manifest_rows, strict=True))
    single, ranks, _ = beauty.frozen_assignments(config, users, identities)
    assert [core.canonical_sha256(rank) for rank in ranks] == audit["full_rank_assignment_sha256"]
    assert core.canonical_sha256(single) == sampler["assignment_sha256"]
    capture = args.purpose == "preflight"
    assignment = preflight_rows(manifest_rows, args.sample_budget) if capture else single
    checkpoint = checkpoint_binding(args.adapter, baseline, args.purpose, identities)
    checkpoint_sha = frozen.sha(args.adapter / CHECKPOINT)
    require(capture or checkpoint["run_commit"] == args.run_commit,
            "formal B must use the selected adapter from the same run snapshot")
    require(frozen.sha(args.adapter / CHECKPOINT) == checkpoint_sha,
            "selected B adapter identity changed during verification")
    versions = {package: importlib.metadata.version(package) for package in baseline["versions"]}
    assert versions == baseline["versions"]
    evaluator_sha = frozen.sha(args.config)
    binding = dict(run_commit=args.run_commit, purpose=args.purpose, backend=args.backend,
                   mode=checkpoint["mode"], plan_sha256=plan_sha, evaluation_config_sha256=evaluator_sha,
                   checkpoint_manifest_sha256=checkpoint_sha, adapter_sha256=checkpoint["adapter_sha256"],
                   checkpoint_run_commit=checkpoint["run_commit"], replay_sha256=REPLAY_SHA,
                   user_axis_sha256=config["user_axis_sha256"], sample_budget=args.sample_budget,
                   versions=versions, visible_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    report = dict(status="RUNNING", argv=sys.argv, RUN_COMMIT=args.run_commit, purpose=args.purpose,
                  backend=args.backend, historical_training_commit=beauty.LETTER_RUN,
                  historical_shuffle_run=SOURCE_V2, host=socket.gethostname(), pid=os.getpid(),
                  interpreter=sys.executable, cwd=str(Path.cwd()), utc_start=dt.datetime.now(dt.timezone.utc).isoformat(),
                  versions=versions, cuda_environment={k: os.environ.get(k) for k in config["gpu_environment"]},
                  author_sources=baseline["author_sources"], data_identity=identities,
                  frozen_entry_sha256=frozen.sha(source_file), history_attention=checkpoint,
                  cpu_manifest_sha256=audit["manifest_sha256"], checkpoint_manifest_sha256=checkpoint_sha,
                  prediction_binding=binding, assigned_users=len(assignment), world_size=1, rank=0, local_rank=0,
                  assignment_sha256=core.canonical_sha256(assignment), expected_conditions=CONDITIONS)
    if capture:
        report["preflight_assignment"] = assignment
    evaluator = source.load_evaluator(author)
    token_ids = audit["token_ids"]
    sid_vocab = set(token_ids.values())

    def validate_row(row, i, mode):
        verify_prediction(row, manifest_rows[i], mode, sid_vocab)
        source.audit_prediction(row, users[i], catalog, evaluator)

    with open_store(output, binding, assignment, args.chunk_users, metadata=report) as store:
        if store.completed is not None:
            print(json.dumps({"stage": "verified_complete", "output": str(output), "users": len(assignment)}),
                  flush=True)
            return

        import torch

        def elapsed():
            return time.monotonic() - started

        def guard():
            if elapsed() > args.max_seconds:
                raise TimeoutError("Beauty generation wall-time limit reached")
            mem = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                       if line.startswith("MemAvailable:"))
            disk = shutil.disk_usage(output).free
            report["min_host_free_bytes"] = min(mem, report.get("min_host_free_bytes", mem))
            report["min_output_free_bytes"] = min(disk, report.get("min_output_free_bytes", disk))
            if mem < 40 * 1024**3 or disk < 20 * 1024**3:
                raise RuntimeError("Frozen evaluator host memory/output disk reserve exhausted")

        try:
            assert torch.cuda.device_count() == 1 and int(os.environ.get("WORLD_SIZE", "1")) == 1
            torch.cuda.set_device(0)
            torch.cuda.reset_peak_memory_stats()
            report["gpu"] = dict(name=torch.cuda.get_device_name(0), visible_devices=binding["visible_gpu"],
                                 total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
            guard()
            started_reuse = time.monotonic()
            prior = store.committed(validate_row)
            report["verified_reuse_seconds"] = time.monotonic() - started_reuse
            report["reused_chunks"] = len(prior)
            report["reused_users"] = sum(len(chunk["assignment"]) for chunk in prior)
            store.note(report)
            sys.path.insert(0, str(author))
            from collator import TestCollator
            from data import SeqRecDataset
            from utils import set_seed

            set_seed(42)
            model, tokenizer = load_inference(args.adapter, report, config, source, author, index)
            assert {name: report["adapter_files"][name] for name in checkpoint["files"]} == checkpoint["files"]
            assert report["adapter_files"][CHECKPOINT]["sha256"] == checkpoint_sha
            assert semantic_hash(tokenizer.get_vocab()) == audit["tokenizer_vocab_sha256"]
            assert tokenizer.get_added_vocab() == token_ids
            assert tokenizer_identity(tokenizer, token_ids) == checkpoint["tokenizer_identity"]
            base = model.get_base_model()
            report["attention_runtime"] = _install_inference(base, args.backend, checkpoint["mode"])
            model.base_model_prepare_inputs_for_generation = base.prepare_inputs_for_generation
            settings, report["generation"] = generation_settings(model)
            data_args = SimpleNamespace(dataset="Beauty", data_path=baseline["data-root"], max_his_len=20,
                                        his_sep=", ", index_file=".index.json", add_prefix=False)
            dataset = SeqRecDataset(data_args, mode="test", sample_num=-1)
            dataset.set_prompt(0)
            assert list(dataset.remapped_inters) == [row["user_id"] for row in users]
            assert all(dataset[i]["labels"] == row["target"] for i, row in enumerate(users))
            allowed = dataset.get_prefix_allowed_tokens_fn(tokenizer)
            report["allowed_callback_sha256"] = hashlib.sha256(
                inspect.getsource(dataset.get_prefix_allowed_tokens_fn).encode()).hexdigest()
            assert report["allowed_callback_sha256"] == baseline["allowed_callback_sha256"]
            store.bind_runtime({"generation": report["generation"], "backend": args.backend,
                                "allowed_callback_sha256": report["allowed_callback_sha256"],
                                "adapter_files": report["adapter_files"], "gpu": report["gpu"]})
            collate = TestCollator(data_args, tokenizer)
            store.note(report)
            new_chunks = 0
            with torch.no_grad():
                for ordinal in range(len(prior), len(store.ranges)):
                    with store.chunk(ordinal, validate_row) as write_user:
                        for i in store.ranges[ordinal]:
                            guard()
                            user = users[i]
                            inputs, targets = collate([dataset[i]])
                            seq = dataset.inters[user["user_id"]]
                            assert targets == [user["target"]] == ["".join(index[str(seq[-1])])]
                            groups = [[token_ids[symbol] for symbol in index[str(item)]] for item in seq[:-1][-20:]]
                            raw, mask = inputs["input_ids"][0].tolist(), inputs["attention_mask"][0].tolist()
                            history, conditions = paired_inputs(raw, mask, groups, sid_vocab, manifest_rows[i])
                            mask_tensor = inputs["attention_mask"].to("cuda:0")
                            history_tensor = torch.tensor([history], dtype=torch.bool, device="cuda:0")
                            generated_rows = {}
                            for mode, (ids, evidence) in conditions.items():
                                ids_tensor = torch.tensor([ids], dtype=inputs["input_ids"].dtype, device="cuda:0")
                                candidates, order, seconds, boundary = predict(
                                    model, base, tokenizer, source, allowed, ids_tensor, mask_tensor, history_tensor,
                                    settings, capture)
                                assert boundary["input_sha256"] == evidence["shuffled_sha256"]
                                assert all(boundary[k] == evidence[k]
                                           for k in ("attention_mask_sha256", "history_sid_mask_sha256"))
                                evidence["actual_prefill"] = boundary
                                if capture:
                                    evidence["original_input_ids"] = raw
                                generated_rows[mode] = dict(user, rank=0, prompt_id=0, input_token_ids=ids,
                                    input_nonpadding_tokens=sum(mask), candidates=candidates,
                                    sorted_generation_order=order,
                                    relevance=[int(candidates[j]["sid"] == targets[0]) for j in order],
                                    generation_seconds=seconds, original_callback_verified_all_tokens=True,
                                    shuffle=evidence)
                            write_user(i, generated_rows)
                    new_chunks += 1
                    progress = dict(stage="five_conditions", completed_chunks=ordinal + 1,
                                    completed_users=sum(map(len, store.ranges[:ordinal + 1])),
                                    assigned_users=len(assignment), elapsed_seconds=elapsed())
                    print(json.dumps(progress), flush=True)
                    report["progress"] = progress
                    store.note(report)
                    if new_chunks == args.stop_after_chunks and ordinal + 1 < len(store.ranges):
                        report["status"] = "PREFLIGHT_STOPPED_AFTER_CHUNKS"
                        return
            guard()
            # Every persisted chunk is checked again before composing the frozen evaluator inputs.
            chunks = store.committed(validate_row)
            require(len(chunks) == len(store.ranges), "final generation lacks a committed chunk")
            target_status = "PREFLIGHT_INTERFACE_PASS" if capture else "PASS"
            report["conditions"] = {
                mode: dict(completed_users=len(assignment),
                           generation_seconds=sum(chunk["totals"][mode]["seconds"] for chunk in chunks),
                           identical_to_normal_rows=sum(chunk["totals"][mode]["identical_to_normal_rows"]
                                                        for chunk in chunks))
                for mode in CONDITIONS
            }
            report["generated_this_attempt"] = len(assignment) - report["reused_users"]
            report["committed_chunks"] = len(chunks)
            report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            final_report = dict(report, status=target_status, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            per_condition = {
                mode: dict(final_report, condition=mode, completed_users=len(assignment),
                           elapsed_generation_seconds=report["conditions"][mode]["generation_seconds"],
                           identical_to_normal_rows=report["conditions"][mode]["identical_to_normal_rows"])
                for mode in CONDITIONS
            }
            store.publish(final_report, per_condition, clock=elapsed)
            report["status"] = target_status
        except BaseException as error:
            report.update(status="FAIL", error=dict(type=type(error).__name__, message=str(error),
                                                   traceback=traceback.format_exc()))
            raise
        finally:
            if torch.cuda.is_initialized():
                report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                report["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            report["elapsed_wall_seconds"] = elapsed()
            report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            store.note(report)


def verify_prediction(row, replay, mode, sid_ids):
    evidence = row['shuffle']
    expected = (dict(input_sha256=replay['original_sha256'], input_changed=False, permutation_ids=[])
                if mode == 'normal' else replay['conditions'][mode])
    assert evidence['mode'] == mode
    assert (row['row_idx'], row['user_id'], row['target']) == (replay['row_idx'], replay['user_id'], replay['target'])
    assert evidence['row_idx'] == row['row_idx']
    assert evidence['cpu_manifest_sha256'] == REPLAY_SHA['inputs.jsonl']
    assert (evidence['original_sha256'], evidence['attention_mask_sha256'], evidence['history_group_count']) == (
        replay['original_sha256'], replay['attention_mask_sha256'], replay['visible_history_length'])
    assert (evidence['input_changed'], evidence['shuffled_sha256'], evidence['permutation_ids']) == (
        expected['input_changed'], expected['input_sha256'], expected['permutation_ids'])
    if mode != 'normal':
        assert all(evidence[key] == value for key, value in core.mode_metadata(mode).items())
    nonpad = row['input_nonpadding_tokens']
    ids = row['input_token_ids']
    assert len(ids) == nonpad == replay['encoded_length']
    assert core.canonical_sha256(ids) == expected['input_sha256']
    history = [token in sid_ids for token in ids]  # Only this verified Prompt0 input omits the answer.
    assert sum(history) == 4 * replay['visible_history_length']
    assert core.canonical_sha256(history) == evidence['history_sid_mask_sha256']
    prefill = evidence['actual_prefill']
    assert prefill['input_sha256'] == expected['input_sha256']
    assert all(prefill[key] == evidence[key] for key in ('attention_mask_sha256', 'history_sid_mask_sha256'))
    assert prefill['position_ids_sha256'] == core.canonical_sha256(list(range(len(ids))))
    assert (prefill['cache_length'], prefill['beam_expansion']) == (0, 20)
    assert row['original_callback_verified_all_tokens']
    assert all(c['eos_position'] == 4 and c['generated_token_ids'][4] == 2 for c in row['candidates'])


def preflight_paired(samples, keys, assignment):
    """A bounded interface check using the unchanged formal pairing/draw routine."""
    import numpy as np

    matrices = {}
    for mode, rows in samples.items():
        require([row["row_idx"] for row in rows] == assignment, "paired sample axis mismatch")
        matrices[mode] = np.asarray([[row["metrics"][k] for k in keys] for row in rows], dtype=np.float64)
    deltas = [matrices[right] - matrices[left] for left, right in statistics.COMPARISONS]
    draws, sample_hash = statistics.paired_means(deltas, len(assignment))
    limits = np.percentile(draws, [2.5, 97.5], axis=1, method="linear")
    means = {mode: {key: math.fsum(matrix[:, j]) / len(assignment) for j, key in enumerate(keys)}
             for mode, matrix in matrices.items()}
    comparisons = {}
    for i, (left, right) in enumerate(statistics.COMPARISONS):
        comparisons[f"{right}-{left}"] = [
            dict(metric=key, difference=means[right][key] - means[left][key],
                 interface_percentile_low=float(limits[0, i, j]), interface_percentile_high=float(limits[1, i, j]))
            for j, key in enumerate(keys)
        ]
    return dict(status="PASS_BOUNDED_PAIRING", scope="technical interface check on selected sample only",
                row_indices=assignment, seed=20261113, bootstrap_replicates=5000,
                draw_indices_sha256_little_endian=sample_hash, metrics=means, comparisons=comparisons)


def _metrics_inventory(root, *, sync=False):
    members = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"unexpected evaluation symlink: {path}")
        if path.is_dir():
            continue
        require(path.is_file(), f"unexpected evaluation member: {path}")
        if path == root / "completion.json":
            continue
        if sync:
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        members[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size, "sha256": frozen.sha(path)}
    if sync:
        for directory in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
            fsync_dir(directory)
        fsync_dir(root)
    return members


def verify_completed_metrics(root, generation_root, run_commit):
    root = Path(root)
    generation = verify_final(generation_root)
    require(root.is_dir() and not root.is_symlink(), "evaluation output unavailable")
    completion_path = root / "completion.json"
    require(completion_path.is_file() and not completion_path.is_symlink(), "evaluation completion is absent")
    completion = read(completion_path)
    require(completion.get("schema") == 1 and completion["files"] == _metrics_inventory(root),
            "evaluation completion member mismatch")
    summary = read(root / "summary.json")
    require(summary["status"] == "PASS_V2_EVALUATION" and generation["purpose"] == "formal"
            and generation["RUN_COMMIT"] == run_commit
            and summary["run_commit"] == summary["evaluation_commit"] == run_commit
            and summary["generation_sha256"] == frozen.sha(Path(generation_root) / "final" / "generation.json")
            and summary["generation_completion_sha256"] == frozen.sha(Path(generation_root) / "final" / "completion.json")
            and summary["checkpoint"] == generation["history_attention"],
            "completed evaluation does not bind to this formal generation")
    return summary


def metrics(args):
    """Verify saved preflight output or compute the full frozen metrics and paired statistics."""
    require(not sys.flags.optimize, "Python optimization disables verified evaluation guards")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    started = time.monotonic()
    check_commit(args.evaluation_commit)
    config = read(args.config)
    durable = os.path.lexists(args.predictions / SPEC)
    if durable:
        generation = verify_final(args.predictions)
        prediction_dir = args.predictions / "final"
    else:
        require(args.purpose == "preflight", "formal metrics require the durable prediction publication")
        prediction_dir = args.predictions
        generation = read(prediction_dir / "generation.json")
    baseline = config["baseline"]
    purpose = generation["purpose"]
    preflight = purpose == "preflight"
    assert purpose in ("preflight", "formal") and args.purpose == purpose
    assert generation["status"] == ("PREFLIGHT_INTERFACE_PASS" if preflight else "PASS")
    assert generation["cpu_manifest_sha256"] == REPLAY_SHA["inputs.jsonl"]
    assert generation["allowed_callback_sha256"] == baseline["allowed_callback_sha256"]
    assert generation["model"] == baseline["model_identity"]
    assert generation["history_attention"]["checkpoint_purpose"] == purpose
    assert generation["history_attention"]["mode"] == "bidirectional"
    assert generation["history_attention"]["adapter_sha256"] == generation["adapter_files"]["adapter_model.bin"]["sha256"]
    assert generation["checkpoint_manifest_sha256"] == generation["adapter_files"][CHECKPOINT]["sha256"]
    audit, rows, sampler = read_replay(args.replay_dir, config)
    sample_budget = generation.get("prediction_binding", {}).get("sample_budget")
    if durable:
        assert generation["prediction_binding"]["evaluation_config_sha256"] == frozen.sha(args.config)
    assignment = preflight_rows(rows, sample_budget) if preflight else None
    assignment_sha = core.canonical_sha256(assignment) if preflight else sampler["assignment_sha256"]
    count = len(assignment) if preflight else config["users"]
    assert generation["assignment_sha256"] == assignment_sha and generation["assigned_users"] == count
    sid_ids = set(audit["token_ids"].values())
    original_config = frozen_json(config, Path(config["normal_evaluation_config"]))
    assert original_config["user_axis_sha256"] == config["user_axis_sha256"]
    run_commit = generation["RUN_COMMIT"]
    check_commit(run_commit)
    if durable:
        assert args.evaluation_commit == run_commit
    original_config["original_run_commit"] = run_commit
    source_path, cce_path = (Path(config[k]) for k in ("frozen_metrics", "official_cce"))
    source = beauty.frozen_dependency(config, source_path)
    official = beauty.frozen_dependency(config, cce_path)
    assert frozen.sha(cce_path) == original_config["official_cce_sha256"]
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "evaluation-config.json"
    frozen.write(config_path, original_config)
    ranges = [(count, assignment_sha)]
    generation_source = beauty.frozen_dependency(config, Path(baseline["entry"]))
    author = generation_source.author_source(baseline["source-root"])
    evaluator = generation_source.load_evaluator(author)
    _, catalog, users, identities = generation_source.catalog_and_users(baseline["data-root"])
    assert identities == generation["data_identity"]

    def check_row(row, mode):
        verify_prediction(row, rows[row["row_idx"]], mode, sid_ids)

    samples = {}
    stage_timings = {}
    if preflight:
        from collections import Counter

        sizes = Counter({sid: len(items) for sid, items in catalog.items()})
        frozen.write(output / "unit_tests.json", source.selftest(official))
    else:
        original = source_path.read_text()
        derived = derive_metrics(original)
        (output / "offline_metrics.derived.py").write_text(derived)
        (output / "offline_metrics.diff").write_text("".join(difflib.unified_diff(
            original.splitlines(True), derived.splitlines(True), fromfile=str(source_path), tofile="V2-single-shard")))
        exec(compile(derived, str(source_path), "exec"), source.__dict__)

    for mode in CONDITIONS:
        stage_started = time.monotonic()
        predictions = prediction_dir / mode
        rank = read(predictions / "rank0.json")
        assert rank["condition"] == mode and rank["status"] == generation["status"]
        for key in ("RUN_COMMIT", "history_attention", "adapter_files", "model", "data_identity",
                    "cpu_manifest_sha256", "allowed_callback_sha256", "checkpoint_manifest_sha256"):
            assert rank[key] == generation[key]
        if durable:
            assert rank["prediction_binding"] == generation["prediction_binding"] and rank["backend"] == generation["backend"]
        assert rank["completed_users"] == rank["assigned_users"] == count
        assert rank["assignment_sha256"] == assignment_sha
        path = predictions / PREDICTIONS
        if preflight:
            with path.open() as stream:
                sample = []
                for i, line in zip(assignment, stream, strict=True):
                    row = json.loads(line)
                    assert row["row_idx"] == i and row["target_item_id"] == users[i]["target_item_id"]
                    check_row(row, mode)
                    generation_source.audit_prediction(row, users[i], catalog, evaluator)
                    sids = [row["candidates"][j]["sid"] for j in row["sorted_generation_order"]]
                    values, _ = source.ranking(sids, row["target"])
                    corrected = source.cce(sids, row["target"], sizes)
                    ref = official.compute_cce_metrics(sids, row["target"], sizes, [5, 10, 20])
                    assert all(abs(corrected[k] - ref[k]) <= 1e-14 for k in ref)
                    values.update(corrected)
                    assert set(values) == set(source.KEYS) and len(values) == 16
                    sample.append({"row_idx": i, "target": row["target"], "metrics": values})
            samples[mode] = sample
            frozen.write(output / f"{mode}.json", dict(scope="bounded technical check", condition=mode,
                                                       predictions_sha256=frozen.sha(path), sample=sample))
        else:
            root = output / mode
            root.mkdir()
            stage_a = generation_source.aggregate_predictions([path], users, catalog, evaluator, run_commit=run_commit)
            stage_a.update(RUN_COMMIT=run_commit, history_attention=generation["history_attention"], condition=mode)
            stage_a_path = root / "stage-a.json"
            frozen.write(stage_a_path, stage_a)
            options = SimpleNamespace(test_dir=predictions, output=root / "primary",
                                      stage_a_report=stage_a_path, config=config_path, run_commit=run_commit,
                                      evaluation_commit=args.evaluation_commit, official_cce=cce_path,
                                      data_root=Path(baseline["data-root"]), assignments=ranges,
                                      check_row=lambda row, mode=mode: check_row(row, mode))
            summary = read(source.evaluate(options))
            summary.update(status="PASS_V2_EVALUATION", condition=mode, candidate_count=count * 20, shards=1,
                           history_attention=generation["history_attention"],
                           history_manifest_sha256=generation["checkpoint_manifest_sha256"],
                           evaluator_derived_sha256=hashlib.sha256(derived.encode()).hexdigest())
            frozen.write(root / "summary.json", summary)
        stage_timings[mode] = time.monotonic() - stage_started
    paired_started = time.monotonic()
    if preflight:
        frozen.write(output / "bounded-paired.json", preflight_paired(samples, source.KEYS, assignment))
    else:
        old_argv = sys.argv
        try:
            sys.argv = ["lc_rec.shuffle_v2.statistics", "--output", str(output / "paired"),
                        "--config", str(args.config), *(str(output / mode) for mode in CONDITIONS)]
            statistics.main()
        finally:
            sys.argv = old_argv
    stage_timings["paired"] = time.monotonic() - paired_started
    frozen.write(output / "summary.json", {"status": "PASS_PREFLIGHT_METRICS" if preflight else "PASS_V2_EVALUATION",
        "users_per_condition": count, "condition_count": len(CONDITIONS), "run_commit": run_commit,
        "evaluation_commit": args.evaluation_commit, "generation_sha256": frozen.sha(prediction_dir / "generation.json"),
        "generation_completion_sha256": frozen.sha(prediction_dir / "completion.json") if durable else None,
        "checkpoint": generation["history_attention"], "elapsed_seconds": time.monotonic() - started,
        "stage_seconds": stage_timings})
    atomically_write_json(output / "completion.json", {"schema": 1, "files": _metrics_inventory(output, sync=True)})


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('generate', 'metrics'):
        sub = commands.add_parser(name, allow_abbrev=False)
        sub.add_argument('--config', type=Path, default=beauty.CONFIG)
        sub.add_argument('--replay-dir', type=Path, default=REPLAY_DIR)
        sub.add_argument('--output', type=Path, required=True)
        sub.add_argument('--purpose', required=True, choices=('preflight', 'formal'))
        if name == 'generate':
            sub.add_argument('--adapter', type=Path, required=True)
            sub.add_argument('--run-commit', required=True)
            sub.add_argument('--max-seconds', type=int, required=True)
            sub.add_argument('--backend', choices=('eager', 'sdpa'), default='eager')
            sub.add_argument('--sample-budget', type=int)
            sub.add_argument('--chunk-users', type=int, default=CHUNK_USERS)
            sub.add_argument('--stop-after-chunks', type=int)
            sub.add_argument('--plan', type=Path)
            sub.add_argument('--formal-authorization', type=Path)
            sub.add_argument('--formal-authorization-sha256')
        else:
            sub.add_argument('--predictions', type=Path, required=True)
            sub.add_argument('--evaluation-commit', required=True)
    if sys.flags.optimize:
        parser.error('Python optimization disables the verified replay guards')
    args = parser.parse_args()
    (generate if args.command == 'generate' else metrics)(args)


if __name__ == '__main__':
    main()
