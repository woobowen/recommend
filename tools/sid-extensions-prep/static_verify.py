"""Read-only byte, AST and local-link verification; never import source code."""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import re
import urllib.parse


def verify(repo):
    manifest = json.loads((repo / 'manifests/sid-extensions/SOURCE_MANIFEST.json').read_text())
    errors = []
    native_issues = []
    checked = 0
    syntax = 0
    for entry in manifest['records']:
        p = repo / entry['path']
        if not p.is_file():
            errors.append({'path': entry['path'], 'error': 'missing'})
            continue
        b = p.read_bytes()
        if len(b) != entry['bytes'] or hashlib.sha256(b).hexdigest() != entry['sha256']:
            errors.append({'path': entry['path'], 'error': 'byte identity mismatch'})
        checked += 1
        if p.suffix == '.py' and entry['source_config_audit_set']:
            try:
                ast.parse(b, filename=entry['path'])
                syntax += 1
            except SyntaxError as ex:
                row = {'path': entry['path'], 'error': str(ex), 'status': 'PRESERVED_NOT_EXECUTED'}
                if entry['kind'] == 'PREPARATION_TOOL_NOT_RESEARCH':
                    errors.append(row)
                else:
                    native_issues.append(row)
    required = ['README.md', 'REPORT.md', 'REQUIREMENTS_MATRIX.md', 'CODE_READING_GUIDE.md',
                'KNOWN_ISSUES_AND_NEXT_CHANGES.md', 'PATH_MAPPING.md', 'ENVIRONMENT.md',
                'experiments/shuffle.md', 'experiments/mlp.md', 'experiments/parallel.md',
                'experiments/attention.md', 'review/README.md']
    docs = repo / 'docs/sid-extensions'
    for relative in required:
        if not (docs / relative).is_file():
            errors.append({'path': str((docs / relative).relative_to(repo)), 'error': 'required document missing'})
    links = 0
    for p in [*docs.rglob('*.md'), repo / 'references/wmh-original/README.md']:
        if not p.exists():
            errors.append({'path': str(p), 'error': 'required reference README missing'})
            continue
        for raw in re.findall(r'\]\(([^)]+)\)', p.read_text()):
            raw = raw.strip('<>')
            if re.match(r'^[a-zA-Z]+:', raw) or raw.startswith('#'):
                continue
            target = (p.parent / urllib.parse.unquote(raw.split('#', 1)[0])).resolve()
            links += 1
            if repo not in target.parents or not target.exists():
                errors.append({'path': str(p.relative_to(repo)), 'link': raw, 'error': 'unresolved local link'})
    return {'status': 'PASS' if not errors else 'FAIL', 'checked_files': checked,
            'python_ast_parsed': syntax, 'local_links_checked': links, 'errors': errors,
            'native_syntax_issues_preserved': native_issues,
            'source_set_id_sha256': manifest['source_set_id_sha256'],
            'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'limits': ['No imports of source/project code', 'No Hydra instantiate, loader, model or tensor deserialization',
                       'AST syntax success does not establish runtime compatibility', 'Native known interface issues are documented separately']}


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--repo', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    args = a.parse_args()
    result = verify(args.repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['errors', 'limits']}, ensure_ascii=False))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
