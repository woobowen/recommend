"""Index verified terminal preparation evidence for byte-preserving transfer."""
import argparse
import json
from pathlib import Path
import re

from lc_rec.shuffle_v2.a1_formal import sha
from lc_rec.shuffle_v2.a1_replay import dump


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--jobs', required=True, type=Path)
    parser.add_argument('--include', action='append', default=[])
    args = parser.parse_args()
    assert args.root == Path('/data/bwwu/lc-rec/a1-instruments-v2')
    assert args.jobs == Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs')
    assert len(args.include) == len(set(args.include))
    args.output.mkdir(parents=True, exist_ok=False)
    members = []
    for name in args.include:
        assert re.fullmatch(r'(?:cpu-[a-z-]+|gpu-preflight)-r[0-3]', name)
        job = args.jobs / ('a1-instruments-v2-' + name)
        status = json.loads((job / 'status.json').read_text())
        assert status['ended_at'] and status['cleanup']['verified']
        allowed = {'request.json', 'accepted.json', 'status.json', 'cleanup.json', 'attempt.json',
                   'work.log', 'supervisor.log'}
        for root, label, selected in ((job, 'jobs', sorted(job / filename for filename in allowed if (job / filename).is_file())),
                                      (args.root / name, 'evidence', sorted((args.root / name).rglob('*')))):
            for member in selected:
                assert not member.is_symlink()
                if member.is_dir():
                    continue
                assert member.is_file()
                previous = member.stat()
                digest = sha(member)
                current = member.stat()
                assert (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns) == (
                    previous.st_dev, previous.st_ino, previous.st_size, previous.st_mtime_ns, previous.st_ctime_ns)
                members.append({'archive_path': f'{label}/{name}/{member.relative_to(root).as_posix()}',
                                'server_path': str(member), 'bytes': previous.st_size, 'sha256': digest})
    dump(args.output / 'manifest.json', {'status': 'PASS_TERMINAL_EVIDENCE_INDEX', 'members': members,
                                         'jobs': args.include, 'bytes': sum(x['bytes'] for x in members)})


if __name__ == '__main__':
    main()
