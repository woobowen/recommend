"""Hash-checked G3 Shuffle wrapper over the shared production generation loop.

The earlier LETTER in-memory source patch is preserved in historical commits.
G3 uses explicit production condition parameters; no textual source mutation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from g3_identity import ASSETS, TOKEN_MAPPING_SHA, digest

FROZEN = {'a0': ('a0_beauty/formal_evaluate.py', '6914ef8b6489fbe76d37d2bc4ed6499d9965a969748b79ddfda6c6d607094b13')}


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--model-key',choices=FROZEN,required=True)
    p.add_argument('--shuffle-n',type=int,choices=(0,3,4),required=True)
    p.add_argument('--acceptance',action='store_true')
    p.add_argument('--cpu-audit',required=True)
    p.add_argument('--derived-only',action='store_true')
    a,remainder=p.parse_known_args()
    original=Path(__file__).resolve().parents[1]/FROZEN[a.model_key][0]
    if hashlib.sha256(original.read_bytes()).hexdigest()!=FROZEN[a.model_key][1]:
        raise ValueError('Production generation source differs from the reviewed G3 source')
    audit=json.loads(Path(a.cpu_audit).read_text())
    if audit['status'] not in ('PASS_CPU_FULL_INPUTS','PASS_CPU_PREFLIGHT_BOUND') or audit['model_key']!='a0':
        raise ValueError('G3 Shuffle CPU binding is incomplete')
    if audit['asset_sha256']!=ASSETS or digest(audit['token_ids'])!=TOKEN_MAPPING_SHA:
        raise ValueError('G3 Shuffle CPU assets or token mapping differ')
    condition={0:'off',3:'s3',4:'s4'}[a.shuffle_n]
    if '--condition' in remainder:
        raise ValueError('Condition is set exactly once by the checked wrapper')
    forwarded=[*remainder,'--condition',condition]
    if a.acceptance:
        if a.shuffle_n!=0 or '--preflight-users-per-rank' not in remainder:
            raise ValueError('Four-condition acceptance is bounded PREFLIGHT OFF only')
        # The common loop writes each condition separately from one strict load.
        forwarded=[*remainder,'--preflight-four-conditions']
    if '--output' not in remainder:
        raise ValueError('Output path is required')
    output=Path(remainder[remainder.index('--output')+1]);output.mkdir(parents=True,exist_ok=True)
    rank=__import__('os').environ.get('RANK','0')
    record={'entry':str(original),'sha256':FROZEN[a.model_key][1],
            'cpu_audit_sha256':hashlib.sha256(Path(a.cpu_audit).read_bytes()).hexdigest(),
            'condition':condition,'forwarded_argv':forwarded,'source_mutation':False}
    (output/('entry_binding.rank'+rank+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    if a.derived_only:return 0
    from formal_evaluate import main as production_main
    sys.argv=[str(original),*forwarded]
    return production_main()


if __name__=='__main__':raise SystemExit(main())
