#!/usr/bin/env python3
"""Fetch pinned upstream files into an isolated directory, checking SHA-256."""
import argparse
import hashlib
import urllib.request
from pathlib import Path

COMMIT = '8d0154e28de37dbb6e24871c508ad8ddb1921cda'
HASHES = {'index': 'be3fb890e59b5443824cdebc560ef349191b97f20d3aecad4727de2d6b212c82',
          'inter': '8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5',
          'item': '74aeddf911548aadfcd1dab326c16678ed4ac74457e874da300d63a94bdb2330'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root / 'Beauty'
    root.mkdir(parents=True, exist_ok=True)
    for kind, expected in HASHES.items():
        target = root / f'Beauty.{kind}.json'
        if not target.exists():
            url = f'https://raw.githubusercontent.com/HonghuiBao2000/LETTER/{COMMIT}/data/Beauty/Beauty.{kind}.json'
            with urllib.request.urlopen(url, timeout=90) as response:
                raw = response.read()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError('Upstream data hash mismatch: ' + kind)
            with target.open('xb') as out:
                out.write(raw)
        raw = target.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Existing data hash mismatch: ' + str(target))
        print(kind, len(raw), expected, target)


if __name__ == '__main__':
    main()
