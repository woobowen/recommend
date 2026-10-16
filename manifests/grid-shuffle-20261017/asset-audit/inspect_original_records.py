"""Read-only source/config selection audit; no tensor/model/GPU execution."""
from pathlib import Path
import base64
import csv
import datetime as dt
from decimal import Decimal
import hashlib
import io
import json
import re
import shutil
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
OLD = Path('/home/addaswsw/recommend-support/wmh-a800-audit-20261010')
ORIGINAL = OLD / 'preservation/A800-grid-20261010-v1/restored/A800_GRID'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def val_records(text):
    rows = list(csv.DictReader(io.StringIO(text)))
    vals = [dict(x, physical_line=i+2) for i, x in enumerate(rows) if x.get('val/recall@5')]
    best = max(vals, key=lambda x: Decimal(x['val/recall@5']))
    return {'count': len(vals), 'maximum': best,
            'ties': [x for x in vals if Decimal(x['val/recall@5']) == Decimal(best['val/recall@5'])]}


def main():
    data = json.loads((HERE / 'remote-b-training-records.json').read_text())
    extra = json.loads((HERE / 'remote-b-source-supplements.json').read_text())
    original_bytes = json.loads((HERE / 'remote-b-source-bytes.json').read_text())
    candidates = json.loads((HERE / 'candidate_specs.json').read_text())
    report = {
        'schema_version': 1,
        'task': 'grid-shuffle-20261017',
        'reviewer': '/root/asset_audit',
        'review_time': dt.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'review_scope': 'Independent original records, exact payload identity and selection audit; no model/tensor execution',
        'source_commit_anchor': 'd0bcacfee4bab4276861573f7746405fb63c54f2',
        'rules_explicitly_read': ['/home/addaswsw/recommend/' + n for n in ['AGENTS.md', 'PROJECT_RULES.md', 'REMOTE_EXECUTION.md']],
        'B': {},
        'A': {},
    }
    stage = 'tiger_train_test_val1600_20261010_112259'
    for domain, stamp in [('toys', '20261010_093036'), ('sports', '20261010_093035')]:
        original_root = REPO / 'GRID baseline/experiments' / f'{domain}_rkmeans_qwen_{stamp}'
        record = data[domain]
        source_dir = HERE / 'B-original-records' / domain
        source_manifest = []
        for rel, entry in {**record, **extra[domain]}.items():
            if 'text' not in entry:
                continue
            dest = source_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            raw = original_bytes[domain][rel]
            dest.write_bytes(base64.b64decode(raw['base64']))
            assert sha(dest) == raw['sha256'] and dest.stat().st_size == raw['bytes']
            pointer = original_root / rel
            expected = None
            if pointer.exists():
                match = re.search(r'oid sha256:([0-9a-f]{64})', pointer.read_text())
                expected = match.group(1) if match else sha(pointer)
            identity = sha(dest)
            if expected is not None:
                assert identity == expected, (domain, rel, identity, expected)
            source_manifest.append({'remote_path': entry['path'], 'local_path': str(dest),
                                    'bytes': dest.stat().st_size, 'sha256': identity,
                                    'github_original_lfs_oid_or_blob_sha256': expected,
                                    'identity_match': identity == expected if expected else None})
        selected = json.loads(record[stage + '/stage_result.json']['text'])
        command = json.loads(record[stage + '/command_val1600.json']['text'])
        maximum = val_records(record[stage + '/csv/version_0/metrics.csv']['text'])
        selected_step = int(re.search(r'step=(\d+)', selected['checkpoint']).group(1))
        assert int(maximum['maximum']['step']) + 1 == selected_step
        assert selected_step < selected['global_step']
        assert command['mode'] == 'fresh_model' and command['resume_checkpoint'] is None
        config_text = record[stage + '/.hydra/config.yaml']['text']
        assert 'monitor: val/recall@5' in config_text and 'mode: max' in config_text
        assert 'save_top_k: 1' in config_text and 'accumulate_grad_batches: 16' in config_text
        assert 'val_check_interval: 1600' in config_text
        pair_argv = [x for x in command['argv'] if x.startswith(('data_dir=', 'semantic_id_path=', 'num_hierarchies=', 'ckpt_path='))]
        checkpoint_pointer = original_root / stage / 'checkpoints' / Path(selected['checkpoint']).name
        expected_checkpoint = re.search(r'oid sha256:([0-9a-f]{64})', checkpoint_pointer.read_text()).group(1)
        assert candidates[domain]['checkpoint']['sha256'] == expected_checkpoint
        for kind, rel in [('config', stage + '/.hydra/config.yaml'), ('sid', 'semantic_ids.pt')]:
            expected = re.search(r'oid sha256:([0-9a-f]{64})', (original_root / rel).read_text()).group(1)
            assert candidates[domain][kind]['sha256'] == expected
        report['B'][domain] = {
            'status': 'SOURCE_AND_PAYLOAD_IDENTITY_PASS_PENDING_LIVE_STRICT_SEMANTIC_GATE',
            'assets': candidates[domain],
            'checkpoint_selection': {
                'monitor': 'val/recall@5', 'mode': 'max', 'save_top_k': 1,
                'selected_step': selected_step, 'training_terminal_step': selected['global_step'],
                'selection_original_stage_result': selected,
                'independent_csv_maximum': maximum,
                'not_selected_by_largest_step_or_test_score': True,
                'validation_interval_batches': 1600, 'accumulate_grad_batches': 16,
                'observed_global_step_interval': 100,
            },
            'fresh_model_training': True,
            'source_training_pairing_argv': pair_argv,
            'upstream': json.loads(record['item_embeddings.json']['text']),
            'source_manifest': source_manifest,
            'remaining_gate': ['actual SID axis and item-to-SID semantic pairing',
                               'checkpoint embedded codebooks equal SID by value',
                               'strict state_dict load and actual shared-table architecture',
                               'formal target coverage from this route; no reuse by sample count alone'],
        }

    contexts_path = OLD / 'analysis/a800-mechanisms/historical-parameter-context-v1/HISTORICAL_PARAMETER_RUN_CONTEXTS.jsonl'
    run_index_path = OLD / 'analysis/a800-mechanisms/A800_EXPERIMENT_RUN_INDEX.jsonl'
    runs = [json.loads(l) for l in run_index_path.open()]
    contexts = [json.loads(l) for l in contexts_path.open()]
    original_manifest = {r['relative_path']: r for r in map(json.loads, (OLD / 'CAPTURE_MANIFEST.jsonl').open()) if r['source_label'] == 'A800_GRID'}
    toys_candidates = []
    for c in contexts:
        if c['run_id'] not in ['logs/train/runs/2025-10-29/23-42-04', 'logs/train/runs/2025-11-03/02-31-15', 'logs/train/runs/2025-11-03/02-06-26']:
            continue
        files = []
        for rel in ['.hydra/config.yaml', '.hydra/overrides.yaml', 'train.log', 'csv/version_0/metrics.csv']:
            source_rel = c['run_id'] + '/' + rel
            src = ORIGINAL / source_rel
            dest = HERE / 'A-original-records' / source_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            expected = original_manifest[source_rel]['source_sha256']
            assert sha(dest) == expected
            files.append({'relative_path': source_rel, 'sha256': expected,
                          'bytes': dest.stat().st_size, 'local_path': str(dest)})
        csv_file = ORIGINAL / c['run_id'] / 'csv/version_0/metrics.csv'
        maximum = val_records(csv_file.read_text())
        toys_candidates.append({
            'run_id': c['run_id'], 'checkpoint': c['source'],
            'coarse_parameter_family': c['coarse_parameter_family'],
            'state_dict_key_count': c['state_dict_keys_checked'],
            'original_parameter_witnesses': c['selected_parameters'],
            'original_test_event_count': c['test_event_count'],
            'independent_csv_validation_maximum': maximum,
            'best_checkpoint_log_lines': [x for x in c['train_log_decisive_lines'] if 'Best ckpt path:' in x['text']],
            'historical_python_source': c['historical_python_source'],
            'historical_forward_mask_loss_generate': c['historical_forward_mask_loss_generate'],
            'source_files': files,
            'eligible_complete_candidate': c['test_event_count'] > 0,
        })
    sid_rel = 'logs/inference/runs/2025-09-17/21-57-54/pickle/merged_predictions_tensor.pt'
    report['A']['toys'] = {
        'status': 'IMPORTANT_CHECKPOINT_SELECTION_UNRESOLVED_NOT_MISSING_ALL_ASSETS',
        'candidates': toys_candidates,
        'shared_declared_sid': {k: original_manifest[sid_rel][k] for k in ['relative_path', 'source_path', 'source_sha256', 'bytes', 'entry_id']},
        'source_model': {'path': 'references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py',
                         'sha256': sha(REPO / 'references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py')},
        'candidate_count_with_completed_original_test': sum(x['eligible_complete_candidate'] for x in toys_candidates),
        'selection_limit': 'Two complete same-SID double-table candidates; neither latest date, largest step nor test score is an approved between-run selection rule. Do not silently choose.',
        'upstream_config_conflict': 'Saved inference config declares Qwen path/3584 while preserved tensor/quantizer identity was independently established as 512-dimensional in prior A800 audit. Do not claim a uniquely bound historical producer.',
        'precise_admission_gaps': ['No approved rule uniquely selecting one of the two completed same-SID original training runs', 'Saved upstream Qwen/3584 declaration conflicts with preserved 512-dimensional content and does not uniquely establish this task T5 producer', 'No current-task strict load/value pairing receipt for a selected Toys A pair'],
        'preferred_next_action': 'Ask Wang senior to identify the intended complete Toys A run and confirm its true upstream T5/RK source chain, then perform only targeted restoration and the existing strict pairing gate. No new training, no SID regeneration, no selection from test scores.',
        'required_if_selected': ['targeted server restoration if payload absent', 'actual embedded SID value pairing', 'strict live model identity gate'],
    }
    sports_runs = [x for x in runs if '/sports' in (x.get('declared_data_dir') or '')]
    assert not sports_runs
    report['A']['sports'] = {
        'status': 'MATCHED_SOURCE_SID_CHECKPOINT_CHAIN_NOT_FOUND_IN_CAPTURED_SCOPE',
        'existing_run_context_index': str(run_index_path),
        'existing_run_context_index_sha256': sha(run_index_path),
        'existing_run_context_count': len(runs),
        'sports_declared_data_context_count': len(sports_runs),
        'scope_limit': 'Reuses existing captured A800 run index and parameter context index; no new full archive scan, no claim Sports was never run historically.',
        'impact': 'Sports A Normal/Shuffle3/Shuffle4 and corresponding attention cannot be authorized from a matched captured pair.',
        'best_next_action': 'Provide a complete matching Sports A source/config/SID/selected checkpoint chain; no retraining or mixed assets within current task.',
    }
    report['warnings'] = [
        'Toys existing MLP spec SID file SHA 26bae0... differs from Toys B c6d516...; old metadata indices must not be reused without route-specific semantic checks.',
        'Source and hash checks do not substitute for checkpoint strict load, semantic pairing, full evaluation, attention correctness or scientific conclusions.',
        'Beauty original formal results were inherited and not recomputed.',
    ]
    report['status'] = 'PASS_SCOPED_SOURCE_AUDIT_WITH_A_UNRESOLVED_AND_LIVE_GATES_PENDING'
    report['input_receipts'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in [HERE / 'remote-b-training-records.json', HERE / 'remote-b-source-supplements.json', HERE / 'remote-b-source-bytes.json', HERE / 'candidate_specs.json', contexts_path, run_index_path]]
    report['export_check_correction'] = {'initial_failure': 'read_text normalized original CRLF CSV before export and SHA mismatch correctly failed closed', 'fix': 'Read server original bytes and transfer base64; verify original byte count/SHA and Git LFS OID before analysis', 'research_inputs_changed': False, 'models_or_GPU_executed': False}
    save(HERE / 'asset-source-review.json', report)
    md = ['# Toys/Sports Shuffle 资产及原选模审查', '',
          '审查者 `/root/asset_audit`；只读原件、有效配置、CSV 和 payload SHA。本回执未运行模型或 GPU，不替代正式准入。', '',
          '| 域/路线 | 判定 | 依据与边界 |', '|---|---|---|',
          '| Toys B | 原来源与载荷身份通过；真实严格配对门待核 | val/recall@5 最大值 0.05609932169318199，CSV step6199 对应 selected ckpt6200，训练终点7200 |',
          '| Sports B | 原来源与载荷身份通过；真实严格配对门待核 | val/recall@5 最大值 0.0332883857190609，CSV step9499 对应 selected ckpt9500，训练终点10500 |',
          '| Toys A | 两份完整候选，不能静默选一份 | 10-29/23-42-04 best5300；11-03/02-31-15 best5000。同SID、双独立1024×128表、178 state keys；之间的选择规则未批准 |',
          '| Sports A | 捕获范围未发现配套链 | 既有345保存Hydra上下文无Sports；不宣称历史从未运行 |', '',
          'B 两域实际服务器 SID、config、checkpoint 的新鲜 SHA 与 Git LFS OID 均相符。正式核验应使用 `candidate_specs.json`，训练验证记录与原来源字节在 `B-original-records/`。', '',
          'Toys A 的原配置、覆盖、train.log、CSV 与哈希收于 `A-original-records/`；保存配置中的 Qwen/3584 与既有实际512载荷冲突必须继续披露。原历史 Python 与完整生成 producer 仍未知，不能据参数形状恢复历史。', '',
          '本轮仅完成来源审查：待主控保存 runner 门验证 checkpoint 内 SID 数值、item轴和严格加载后可补充正式准入结论。详细逐条证据见 `asset-source-review.json`。', '']
    (HERE / 'asset-source-review.md').write_text('\n'.join(md))
    print(json.dumps({'status': report['status'], 'report': str(HERE / 'asset-source-review.json'), 'sha256': sha(HERE / 'asset-source-review.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
