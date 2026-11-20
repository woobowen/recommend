"""Explicit V2 adapter over the byte-identical frozen Instruments entry sources."""
import argparse
import difflib
import hashlib
import os
from pathlib import Path
import sys

from lc_rec.shuffle.entry import FROZEN, adapt
from lc_rec.shuffle_v2.core import MODES
from lc_rec.shuffle_v2.runtime import Runtime


def derive(text, model_key):
    derived = adapt(text, model_key)
    if model_key == 'a1':
        before = 'from runtime import ShuffleAllowed as Allowed'
        assert derived.count(before) == 1
        derived = derived.replace(before, 'Allowed = runtime.allowed')
    if model_key == 'a0':
        replacements = (
            ('args.preflight_users_per_rank not in (0, 16)', 'args.preflight_users_per_rank not in (0, 32)'),
            ('separate fixed eight-user/rank preflight', 'separate fixed 32-user single-rank preflight'),
            ('if world != 2 or not 0 <= rank < 2', 'if world != 1 or rank != 0 or local_rank != 0'),
            ('Requires two ranks, at least', 'Requires rank0/local0, at least'),
            ('len(assignment) != 12386', 'len(assignment) != 24772'),
            ('range(2)],\n                                       users, catalog', 'range(1)],\n                                       users, catalog'),
        )
        for old, new in replacements:
            assert derived.count(old) == 1, old
            derived = derived.replace(old, new)
    return derived


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model-key', choices=FROZEN, required=True)
    parser.add_argument('--shuffle-mode', choices=MODES, required=True)
    parser.add_argument('--cpu-audit', type=Path, required=True)
    parser.add_argument('--single-shard-audit', type=Path)
    parser.add_argument('--derived-only', action='store_true')
    options, remainder = parser.parse_known_args()
    if any(k.startswith('--aggregate') for k in remainder):
        parser.error('Use the CPU evaluation entry for saved predictions')
    original, sha = FROZEN[options.model_key]
    source = Path(__file__).resolve().parents[1] / original
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sha
    text = raw.decode('utf-8')
    derived = derive(text, options.model_key)
    bytecode = compile(derived, str(source), 'exec')
    output = Path(remainder[remainder.index('--output') + 1])
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get('RANK', '0'))
    with (output / f'derived.rank{rank}.py').open('x') as stream:
        stream.write(derived)
    with (output / f'derived.rank{rank}.diff').open('x') as stream:
        stream.writelines(difflib.unified_diff(text.splitlines(True), derived.splitlines(True),
                                              fromfile=str(source), tofile='V2-derived'))
    if options.derived_only:
        return 0
    runtime = Runtime(options)
    namespace = {'__name__': 'shuffle_v2_frozen_entry', '__file__': str(source), 'runtime': runtime}
    exec(bytecode, namespace)
    sys.argv = [str(source), *remainder]
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
