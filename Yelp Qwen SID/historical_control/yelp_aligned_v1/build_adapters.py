"""Create dataset-only adapters with reviewable diffs; original entries untouched."""
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


qwen = (HERE/'original_generate_qwen_embeddings.py').read_text(encoding='utf-8')
qnew = replace_once(qwen, 'parents[1]', 'parents[2]')
qnew = replace_once(qnew, "choices=['beauty', 'sports', 'toys']", "choices=['yelp_aligned_v1']")
qnew = replace_once(qnew, "    parser.add_argument('--output'", "    parser.add_argument('--data-dir', type=Path, required=True)\n    parser.add_argument('--output'")
qnew = replace_once(qnew, "ROOT/'original_assets/data/amazon_data'/args.dataset/'items'", "args.data_dir/'items'")
(HERE/'generate_qwen_embeddings.py').write_text(qnew, encoding='utf-8')

runner = (HERE/'original_run_task.py').read_text(encoding='utf-8')
rnew = replace_once(runner, "BASELINES = {'beauty': 'beauty_rkmeans_qwen_20261010_012253',\n             'toys': 'toys_rkmeans_qwen_20261010_093036'}", "BASELINES = {'yelp_aligned_v1': 'yelp_rkmeans_qwen_aligned_v1'}\nDATA_DIR = Path('/home/jcy/data/yelp_aligned_v1')")
rnew = replace_once(rnew, "    ap.add_argument('--preflight', action='store_true')", "    ap.add_argument('--preflight', action='store_true')\n    ap.add_argument('--prepare-only', action='store_true', help='Verify loaders/model/samples without fit or test')")
rnew = replace_once(rnew, "ROOT / f'original_assets/data/amazon_data/{args.dataset}'", 'DATA_DIR')
rnew = replace_once(rnew, 'str(path.relative_to(ROOT)): digest(path) for path in raw_files', 'str(path): digest(path) for path in raw_files')
rnew = replace_once(rnew, "list((ROOT/'codex/plan_v1_1').glob('*.py'))", "list((ROOT/'codex/plan_v1_1').glob('*.py')) + list((ROOT/'codex/yelp_aligned_v1').glob('*.py'))")
rnew = replace_once(rnew, "f'data_dir={ROOT}/original_assets/data/amazon_data/{args.dataset}'", "f'data_dir={DATA_DIR}'")
rnew = replace_once(rnew, '        assert cfg.train and cfg.test\n        p.trainer.fit', '''        assert cfg.train and cfg.test
        if args.prepare_only:
            from codex.yelp_aligned_v1.preflight import inspect_without_training
            report = inspect_without_training(p, cfg, args.variant)
            report.update(fit_called=False, test_called=False, formal_training_started=False,
                          frozen_users=len(samples), sample_manifest_sha256=digest(out/'sample_manifest.json'))
            write_json(out/'preparation_verified.json', report)
            print(json.dumps(report, indent=2), flush=True)
            return
        p.trainer.fit''')
(HERE/'run_task.py').write_text(rnew, encoding='utf-8')
for name, before, after in [('generate_qwen_embeddings',qwen,qnew),('run_task',runner,rnew)]:
    (HERE/f'{name}.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
        fromfile=f'original/{name}.py', tofile=f'yelp_aligned_v1/{name}.py')), encoding='utf-8')
manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('original_*.py')}
(HERE/'adapter_sources.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
