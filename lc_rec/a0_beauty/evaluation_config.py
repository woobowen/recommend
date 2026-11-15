#!/usr/bin/env python3
"""Derive Beauty evaluation identity without changing ranking definitions."""
import argparse
import json
from pathlib import Path

ASSETS = {'index': 'be3fb890e59b5443824cdebc560ef349191b97f20d3aecad4727de2d6b212c82',
          'inter': '8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5',
          'item': '74aeddf911548aadfcd1dab326c16678ed4ac74457e874da300d63a94bdb2330'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--run-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert len(args.run_commit) == 40 and all(c in '0123456789abcdef' for c in args.run_commit)
    cfg = json.loads(args.baseline.read_text())
    assert cfg['protocol_version'] == 'letter_instruments_beam20_v1'
    cfg.update(protocol_version='letter_beauty_beam20_v1', dataset='Beauty', users=22363,
               user_axis='insertion order of frozen Beauty.inter.json',
               original_run_commit=args.run_commit, sid_coverage_denominator=12088,
               asset_sha256=ASSETS,
               user_axis_sha256='7430079e2b917dfa5e4bb8ead2702fb6c622fb48ffb684dd2889780f9480af7f')
    with args.output.open('x') as out:
        json.dump(cfg, out, indent=2)
        out.write('\n')


if __name__ == '__main__':
    main()
