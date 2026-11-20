"""Separate adapters for the frozen LETTER and G3 Beauty generation entries."""
import argparse
import difflib
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

from lc_rec.shuffle_v2 import core
from lc_rec.shuffle_v2.replay import dump
from lc_rec.shuffle_v2.runtime import Runtime

LETTER_RUN = '6e1b211d9c37f358c43b1540b40b17f6703150d2'
G3_RUN = '21d495b0be5b606853ec7cec68a3eaa1b58acbd7'
SHA = {
    'letter': '998c5b3201954d9c72fa71aeb5ba6e7473dc6ce6e5fcce342889f6fde4daf20f',
    'g3': '9c7748ce7659a085d15128e060a1b72f6ac92b62a8642056810d0fa6df6a14e7',
    'letter_shuffle_entry': '71d80574f439f0f66cfbd5051e65e293d19415c950137468c98e63e916a7da17',
}


def read_frozen(path, key):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA[key], path
    return raw.decode('utf-8')


def derive_letter(text, entry):
    read_frozen(entry, 'letter_shuffle_entry')
    spec = importlib.util.spec_from_file_location('frozen_letter_beauty_v1_adapter', entry)
    frozen_adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen_adapter)
    anchor = '            assert assignment == runtime.preflight_assignment(rank)'
    derived = frozen_adapter.adapt(text, 'a0')
    assert derived.count(anchor) == 1
    return derived


def g3_transform(raw, mask, groups, vocab, mode, row_idx):
    actual, evidence = core.transform(raw, mask, groups, vocab, mode, row_idx)
    return actual, {**core.mode_metadata(mode), 'row_idx': row_idx, 'original_input_token_ids': list(raw),
                    'attention_mask': list(mask), 'evidence': evidence}


def derive_g3(text):
    def change(old, new, times=1):
        nonlocal text
        assert text.count(old) == times, old
        text = text.replace(old, new)

    change('from shuffle_runtime import transform_inputs', 'transform_inputs = v2_transform_inputs', 2)
    change("n = {'normal': None, 'off': 0, 's3': 3, 's4': 4}[condition]", 'n = condition')
    change("        n = {'off':0,'s3':3,'s4':4}[condition]\n        if record['n'] != n or record['seed'] != 42 or record['protocol'] != 'GRID-LETTER-SHUFFLE-V1':",
           "        n = condition\n        if any(record.get(k) != v for k, v in v2_mode_metadata(n).items()):")
    change('choices=("normal", "off", "s3", "s4"), default="normal"', 'choices=v2_modes, required=True')
    change("{'n':{'off':0,'s3':3,'s4':4}[args.condition],\n                                 'seed':42,'protocol':'GRID-LETTER-SHUFFLE-V1'}", 'v2_mode_metadata(args.condition)')
    change('{"n": {"off":0,"s3":3,"s4":4}[condition],\n                       "seed":42,"protocol":"GRID-LETTER-SHUFFLE-V1"}', 'v2_mode_metadata(condition)')
    return text


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--source', type=Path, required=True, help='Actual frozen Beauty formal_evaluate.py')
    parser.add_argument('--variant', choices=('letter', 'g3'), required=True)
    parser.add_argument('--shuffle-mode', choices=core.MODES, required=True)
    parser.add_argument('--v2-run-commit', required=True)
    parser.add_argument('--cpu-audit', type=Path)
    parser.add_argument('--derived-only', action='store_true')
    options, remainder = parser.parse_known_args()
    allowed = {'--source-root', '--data-root', '--model', '--adapter', '--output', '--max-seconds',
               '--minimum-disk-free-gib', '--minimum-host-free-gib', '--preflight-users-per-rank', '--preflight-manifest'}
    assert len(remainder) % 2 == 0 and set(remainder[::2]) <= allowed
    assert len(set(remainder[::2])) == len(remainder[::2]) and '--output' in remainder
    assert len(options.v2_run_commit) == 40 and all(c in '0123456789abcdef' for c in options.v2_run_commit)
    source = options.source.resolve()
    text = read_frozen(source, options.variant)
    output = Path(remainder[remainder.index('--output') + 1])
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get('RANK', '0'))
    binding = {**core.mode_metadata(options.shuffle_mode), 'V2_RUN_COMMIT': options.v2_run_commit,
               'frozen_source': str(source), 'frozen_source_sha256': SHA[options.variant]}
    namespace = {'__name__': 'shuffle_v2_beauty_frozen_entry', '__file__': str(source)}
    if options.variant == 'letter':
        derived = derive_letter(text, source.with_name('shuffle_entry.py'))
        argv = [*remainder, '--run-commit', options.v2_run_commit]
        if not options.derived_only:
            options.model_key = 'beauty_letter'
            namespace['runtime'] = Runtime(options)
    else:
        # G3 loader's run-commit parameter validates the existing training adapter's
        # provenance. The binding above separately identifies the V2 code deployed.
        argv = [*remainder, '--run-commit', G3_RUN, '--condition', options.shuffle_mode]
        derived = derive_g3(text)
        namespace.update(v2_transform_inputs=g3_transform, v2_mode_metadata=core.mode_metadata, v2_modes=tuple(core.MODES))
    binding['effective_argv'] = argv
    binding['derived_sha256'] = hashlib.sha256(derived.encode('utf-8')).hexdigest()
    dump(output / f'v2-binding.rank{rank}.json', binding)
    with (output / f'derived.rank{rank}.py').open('x') as stream:
        stream.write(derived)
    with (output / f'derived.rank{rank}.diff').open('x') as stream:
        stream.writelines(difflib.unified_diff(text.splitlines(True), derived.splitlines(True), fromfile=str(source), tofile='V2-derived'))
    code = compile(derived, str(source), 'exec')
    if options.derived_only:
        return 0
    sys.path.insert(0, str(source.parent))
    sys.argv = [str(source), *argv]
    exec(code, namespace)
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
