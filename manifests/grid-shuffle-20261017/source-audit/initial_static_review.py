"""Read-only source facts for the independent Toys/Sports Shuffle review."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
OUT = Path(__file__).parent / 'initial-static-review.json'
NAMES = [
    'experiments/formal_20261015/run.py',
    'experiments/formal_20261015/deploy.py',
    'experiments/beauty_extensions/common.py',
    'experiments/beauty_extensions/authorization.py',
    'experiments/beauty_extensions/baseline.py',
    'experiments/beauty_extensions/author/models.py',
    'experiments/beauty_extensions/results.py',
    'experiments/beauty_extensions/data_checks.py',
    'experiments/beauty_extensions/cli.py',
    'experiments/beauty_extensions/observation.py',
    'experiments/beauty_extensions/attention/capture.py',
    'experiments/beauty_extensions/attention/records.py',
    'experiments/beauty_extensions/attention/plot.py',
    'experiments/preflight_20261015/run.py',
    'experiments/beauty_extensions/mlp_formal.py',
    'GRID baseline/src/models/modules/semantic_id/tiger_generation_model.py',
    'GRID baseline/src/components/eval_metrics.py',
    'references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py',
    'manifests/grid-formal-20261015/attention_statistics.py',
    'manifests/grid-formal-20261015/attention-results-review/verify_attention_statistics.py',
    'manifests/grid-formal-20261015/attention-results-review/verify_attention_exports.py',
]


def identity(path):
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


files = [identity(ROOT / name) for name in NAMES]
for name in NAMES:
    ast.parse((ROOT / name).read_bytes(), filename=name)

run = (ROOT / NAMES[0]).read_text()
hardcoded = [i for i, text in enumerate(run.splitlines(), 1) if '22363' in text]
result = {
    'status': 'STATIC_REVIEW_COMPLETE_RUNTIME_ADMISSION_PENDING',
    'reviewer_task': '/root/source_audit',
    'reviewed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'rule_sources': [identity(Path('/home/addaswsw/recommend') / name)
                     for name in ('AGENTS.md', 'PROJECT_RULES.md', 'REMOTE_EXECUTION.md')],
    'source_files': files,
    'python_syntax_files_checked': len(files),
    'needed_minimal_changes': [
        {'file': NAMES[0], 'lines': hardcoded,
         'change': 'Replace both Beauty 22363 assertions with the number of targets in the hash-verified current-route test_index using existing read_index. Retain exact target Counter coverage comparison; reject an empty index.',
         'research_impact': 'Coverage validation only; original model, SID, checkpoint, loader, RNG, generation and evaluator remain unchanged.'},
        {'file': NAMES[1],
         'change': 'Use a task-specific thin deployment wrapper or parameterization for support root, remote root and job prefix; keep the accepted detached runner and immutable snapshot/source checks.',
         'research_impact': 'Output/request isolation only.'},
        {'file': 'manifests/grid-formal-20261015/attention-results-review/verify_attention_exports.py',
         'change': 'New-task verifier wrapper must parameterize report destination, reviewer identity and attention spec path; existing verifier currently constructs attention-beauty-*.json.',
         'research_impact': 'Evidence navigation only, preserve all matrix/head/mask/beam/pixel checks.'},
        {'file': 'manifests/grid-formal-20261015/attention-results-review/verify_attention_statistics.py',
         'change': 'New-task independent verifier wrapper must redirect SUPPORT/output/reviewer identity; keep pair_values, summaries and least-three formula unchanged.',
         'research_impact': 'Evidence output isolation only.'},
    ],
    'reviewed_mechanism': {
        'B_hook_position': 'Shared SID lookup invoked inside encoder_forward_pass after offset and before SEP. Out-of-encoder lookup (including decoder) returns untouched output.',
        'normal_rng': 'shuffle_n==0 branch returns before Generator construction or torch.rand; native model computation and original seeded dropout protocol remain inherited.',
        'sampler': 'Frozen original torch.rand(B,M,N).argsort(2), independently indexed by occurrence; torch.where leaves padding groups unchanged; Shuffle3 leaves coordinate 4 unchanged.',
        'trace_rng': 'Permutation trace uses a separate generator initialized from native RNG state and never resets/advances the global stream. Formal trace is bounded as in accepted Beauty, with full role/call counters.',
        'test_lifecycle': 'Original eval_step computes loss then generate; each invokes encoder, therefore each has its own native permutation. Existing smoke/shuffle_check checks role deltas and fault recovery.',
        'strict_loading': 'verify_asset rejects LFS pointers and hash mismatch; load_state_dict(strict=True), exact key/shape/value equality, alias conflict checks and embedded SID equality are already implemented.',
        'A_identity': 'Author model uses separate encoder/decoder embedding objects and original author Shuffle helper; cannot load B shared-table weights into it.',
        'data_and_cohort': 'Data checks derive actual SID item_count and all target/mask metadata from selected cfg+spec, no fixed Beauty item_count. Must create cohort for each actual B SID rather than reuse different-SID MLP cohort.',
        'attention': 'Observer requests already-computed eager last-layer cross-attention; wraps unchanged beam method once; reconstructs unique exact prefixes and final ancestry, keeps all heads and both independent path types.',
        'attention_off_on': 'capture_batch holds identical initial CPU/CUDA RNG, requires exact generated IDs/scores and final RNG equality, parameter digest and hook restoration, exception-cleanup probe.',
        'figures': 'Raw head-mean values, complete encoder positions, per-figure common scale; no row normalization or clipping; one figure per fixed sample.',
        'statistics': 'Primary statistics program accepts a plan of job directories and has no domain-specific formula. Cosine/Pearson raw visible weights; probability-only normalization for TV/JS; two paths kept separate.',
        'key_compatibility': 'Retain beauty-extensions-v1 opaque key prefix and schema names for compatibility; these are version names, not selection filters. Use domain context when comparing users across domains.',
    },
    'not_claimed': ['No real Toys/Sports checkpoint/SID was loaded by this reviewer.',
                    'No GPU, remote experiment or formal gate was launched by this reviewer.',
                    'Runtime admission, source diff regression and deployed-byte identity remain pending.'],
    'new_GPU_or_model_calls': 0,
    'source_or_Git_writes': 0,
    'review_program': identity(Path(__file__)),
}
assert not OUT.exists(), 'Preserve this review identity; do not overwrite it'
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print(json.dumps(identity(OUT) | {'status': result['status'], 'syntax_files': len(files)}))
