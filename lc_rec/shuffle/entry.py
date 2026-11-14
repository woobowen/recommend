"""Apply a small, hash-checked intervention to the frozen Normal entry in memory.

Original source stays untouched. The exact derived source and unified diff are
saved before execution; model reload and generation arguments remain original.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import sys

FROZEN = {
    'a0': ('letter/scripts/formal_evaluate.py', '47aa87e2d4b1073e076a3d76ef8af07682cb45fc9dd8c3fb09113e1bb4d000d7'),
    'a1': ('a1_qwen3_06b/scripts/evaluate.py', 'c08f6cceb982147009e6d602a6b693065830e33a5b24e7d40b1d79b7156981f1'),
}


def adapt(text, model):
    def change(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError('Frozen source anchor is not unique: ' + old[:80])
        text = text.replace(old, new)
    change('args.preflight_users_per_rank not in (0, 8)',
           'args.preflight_users_per_rank not in (0, 16)')
    if model == 'a1':
        begin = text.index('        if args.preflight_users_per_rank:\n            # Fixed raw-text')
        end = text.index('        loader = DataLoader', begin)
        text = text[:begin] + '''        if args.preflight_users_per_rank:
            assignment = runtime.preflight_assignment(rank)
            report["preflight_assignment"] = assignment
''' + text[end:]
        change('from common import Allowed', 'from runtime import ShuffleAllowed as Allowed')
        change('allowed.bind(input_ids, data[index]["input_ids"])',
               'allowed.bind(input_ids, data[index]["input_ids"], shuffle_record)')
    else:
        change('        loader = DataLoader(indexed, batch_size=1, sampler=sampler, collate_fn=collate,',
               '''        if args.preflight_users_per_rank:
            assignment = runtime.preflight_assignment(rank)
            report["preflight_assignment"] = assignment
        loader = DataLoader(indexed, batch_size=1, sampler=assignment, collate_fn=collate,''')
    change('                input_ids = inputs["input_ids"][0].tolist()',
           '''                inputs, shuffle_record = runtime.intervene(inputs, index, data, tokenizer)
                input_ids = inputs["input_ids"][0].tolist()''')
    change('"input_token_ids": input_ids,',
           '"input_token_ids": input_ids, "shuffle": shuffle_record,')
    change('model, tokenizer = strict_reload(args, report, torch, local_rank, source, indices)',
           '''model, tokenizer = strict_reload(args, report, torch, local_rank, source, indices)
        runtime.loaded(model, tokenizer, args, report)''')
    return text


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--model-key', choices=FROZEN, required=True)
    p.add_argument('--shuffle-n', type=int, choices=[0, 3, 4], required=True)
    p.add_argument('--acceptance', action='store_true')
    p.add_argument('--cpu-audit', required=True)
    p.add_argument('--derived-only', action='store_true')
    a, remainder = p.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    original = root / FROZEN[a.model_key][0]
    raw = original.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FROZEN[a.model_key][1]
    text = raw.decode()
    derived = adapt(text, a.model_key)
    assert '--output' in remainder
    out = Path(remainder[remainder.index('--output') + 1])
    out.mkdir(parents=True, exist_ok=True)
    rank = __import__('os').environ.get('RANK', '0')
    (out / ('derived.rank' + rank + '.py')).write_text(derived)
    (out / ('derived.rank' + rank + '.diff')).write_text(''.join(difflib.unified_diff(
        text.splitlines(True), derived.splitlines(True), fromfile=str(original), tofile='derived')))
    compile(derived, str(original), 'exec')
    if a.derived_only:
        return
    import runtime
    runtime.configure(a)
    namespace = {'__file__': str(original), '__name__': 'shuffle_frozen_entry', 'runtime': runtime}
    exec(compile(derived, str(original), 'exec'), namespace)
    sys.argv = [str(original), *remainder]
    raise SystemExit(namespace['main']())


if __name__ == '__main__':
    main()
