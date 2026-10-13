"""Build byte/symbol/import/config indexes by reading files; never import them."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.parse

R = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
S = Path('/home/addaswsw/recommend-support/sid-extensions-p0')
B = '563dc50b1c7e9e2e05cdfddd13e91a3d1888001e'
M = R / 'manifests/sid-extensions'


def identity(p):
    b = p.read_bytes()
    return {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}


def write(name, data):
    (M / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    records = []
    symbols = []
    imports = []
    targets = []
    for line in subprocess.check_output(['git', 'ls-tree', '-r', '-z', B], cwd=R).split(b'\0'):
        if not line:
            continue
        meta, rawpath = line.split(b'\t', 1)
        mode, kind, blob = meta.decode().split()
        path = rawpath.decode()
        if not path.startswith(('GRID baseline/', 'Baseline Results/')):
            continue
        p = R / path
        b = p.read_bytes()
        expected_blob = hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()
        assert expected_blob == blob, 'baseline checkout bytes changed: ' + path
        core = (path.startswith(('GRID baseline/src/', 'GRID baseline/configs/', 'GRID baseline/_运行/', 'GRID baseline/codex/'))
                or path.startswith('Baseline Results/') or p.name in ['LICENSE', 'requirements.txt', 'README.md', '.project-root', '.gitattributes', '__init__.py'])
        row = {'path': path, 'kind': 'BASELINE_INHERITED_BYTE_IDENTICAL', 'baseline_sha': B,
               'git_blob': blob, 'git_mode': mode, **identity(p), 'source_config_audit_set': core,
               'server_path': '/home/bwwu/recommend-worktrees/sid-extensions-p0/' + path}
        match = re.fullmatch(rb'version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n', b)
        if match:
            row.update(storage='GIT_LFS_POINTER', lfs_oid='sha256:' + match[1].decode(), payload_bytes=int(match[2]), source_config_audit_set=False)
        else:
            row['storage'] = 'ORDINARY_GIT_BLOB'
        records.append(row)
    restore = json.loads((S / 'jobs/new-server/sid-extensions-p0-restore-v1/retrieved/output/RESULT.json').read_text())
    plan = json.loads((S / 'requests/restore-selected-v1-plan.json').read_text())
    batches = {e['batch']: e for e in plan['batches']}
    for e in restore['rows']:
        if e['purpose'] != 'original_source_or_historical_config':
            continue
        p = R / e['logical_path']
        assert identity(p) == {'bytes': e['bytes'], 'sha256': e['sha256']}
        records.append({'path': e['logical_path'], 'kind': 'WMH_NATIVE_ARCHIVE_ORIGINAL',
            'storage': 'ORDINARY_GIT_BLOB', 'source_config_audit_set': True, **identity(p),
            'capture_version': e['capture_version'], 'original_source_path': e['source_path'],
            'archive_member': e['member_path'], 'archive': batches[e['batch']], 'entry_id': e['entry_id'],
            'server_original_copy': e['actual_path'],
            'server_path': '/home/bwwu/recommend-worktrees/sid-extensions-p0/' + e['logical_path'],
            'server_original_copy_sha256': e['actual_sha256'], 'redaction': 'NONE_BYTE_IDENTICAL',
            'research_author': '王旻镐 / original GRID authors; see preserved LICENSE and source context'})
    small = json.loads((S / 'evidence/local-run-records.json').read_text())
    for e in small:
        path = 'manifests/sid-extensions/baseline-run-records/' + e['dataset'] + '/' + e['repository_path'].split('GRID baseline/', 1)[1]
        assert identity(R / path) == {'bytes': e['bytes'], 'sha256': e['sha256']}
        records.append({'path': path, 'kind': 'BASELINE_LFS_TEXT_PAYLOAD_READABLE_COPY', 'storage': 'ORDINARY_GIT_BLOB',
            'source_config_audit_set': True, **identity(R / path), 'baseline_sha': B,
            'source_path': e['repository_path'], 'lfs_oid': e['lfs_oid'], 'dataset': e['dataset'],
            'server_path': '/home/bwwu/recommend-worktrees/sid-extensions-p0/' + path,
            'server_payload_path': e['destination'], 'redaction': 'NONE_BYTE_IDENTICAL'})
    for p in sorted((R / 'tools/sid-extensions-prep').glob('*.py')):
        records.append({'path': str(p.relative_to(R)), 'kind': 'PREPARATION_TOOL_NOT_RESEARCH',
            'storage': 'ORDINARY_GIT_BLOB', 'source_config_audit_set': True, **identity(p),
            'server_path': '/home/bwwu/recommend-worktrees/sid-extensions-p0/' + str(p.relative_to(R))})
    for e in records:
        if not e['source_config_audit_set']:
            continue
        p = R / e['path']
        if p.suffix == '.py':
            try:
                tree = ast.parse(p.read_bytes())
            except SyntaxError as ex:
                symbols.append({'path': e['path'], 'sha256': e['sha256'], 'syntax': 'NATIVE_SYNTAX_ISSUE_PRESERVED', 'error': str(ex)})
                continue
            def visit(nodes, parent=''):
                for n in nodes:
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        name = parent + n.name
                        symbols.append({'path': e['path'], 'sha256': e['sha256'], 'symbol': name,
                                        'start': n.lineno, 'end': n.end_lineno, 'kind': type(n).__name__})
                        visit(n.body, name + '.')
            visit(tree.body)
            for n in ast.walk(tree):
                mods = []
                if isinstance(n, ast.Import):
                    mods = [x.name for x in n.names]
                elif isinstance(n, ast.ImportFrom) and n.module:
                    mods = [n.module]
                for mod in mods:
                    root = mod.split('.')[0]
                    if root == 'src':
                        source_root = R / ('references/wmh-original/GRID' if e['path'].startswith('references/') else 'GRID baseline')
                        target = source_root.joinpath(*mod.split('.'))
                        found = target.with_suffix('.py').is_file() or (target / '__init__.py').is_file() or target.is_dir()
                        category = 'LOCAL_MODULE_PATH_FOUND' if found else 'LOCAL_MODULE_PATH_UNRESOLVED'
                    else:
                        category = 'STDLIB' if root in sys.stdlib_module_names else 'THIRD_PARTY_OR_TASK_TOOL'
                    imports.append({'path': e['path'], 'line': n.lineno, 'module': mod, 'category': category,
                                    'limit': 'Static module path; exported names, runtime dispatch and import side effects not executed.'})
        elif p.suffix in ['.yaml', '.yml']:
            for i, line in enumerate(p.read_text().splitlines(), 1):
                match = re.search(r'_target_:\s*(\S+)', line)
                if match:
                    target = match[1].strip('"\'')
                    targets.append({'path': e['path'], 'line': i, 'target': target,
                                    'nature': 'CONFIG_DECLARATION', 'instantiation': 'NOT_RUN'})
    content = sorted((e['path'], e['sha256']) for e in records if e['source_config_audit_set'])
    set_id = hashlib.sha256(json.dumps(content, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    write('SOURCE_MANIFEST.json', {'stage_id': 'sid-extensions-p0', 'baseline_sha': B,
          'source_set_id_sha256': set_id, 'source_set_definition': 'sorted path/SHA256 pairs for source_config_audit_set=true; excludes manifests/docs to avoid self-reference',
          'original_notice_reference': '27,047,918-byte notices.txt kept in original b01 archive; preserved LICENSE published. No notebook, .git, environment or large archive copied into references.',
          'records': records})
    write('SOURCE_LINES.json', {'source_set_id_sha256': set_id, 'symbols': symbols})
    write('STATIC_DEPENDENCIES.json', {'source_set_id_sha256': set_id, 'imports': imports, 'hydra_targets': targets,
          'limits': ['No source imports or Hydra instantiate', 'Dynamic targets/resolvers, exported symbols and runtime import identities not confirmed', 'No model load or data-loader execution', 'Native interfaces kept unchanged even when statically incompatible']})
    print('catalog:', len(records), 'audited source/config', len(content), 'symbols', len(symbols), 'imports', len(imports), 'set', set_id)


if __name__ == '__main__':
    main()
