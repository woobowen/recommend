"""Build one bounded final review ZIP from fixed Git bytes and terminal evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT = Path('/home/addaswsw/recommend-support/grid-formal-20261015')
JOBS = Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server')
BASE = '93567ea7f61e970c448838443595242d0b9beef0'
FORBIDDEN = {'.ckpt', '.pt', '.pth', '.pkl', '.pyc', '.tfrecord', '.tar', '.gz', '.zip'}
SECRET = re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{30,}')


def git_bytes(commit, path):
    return subprocess.check_output(['git', 'show', commit+':'+path], cwd=REPO)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit', required=True)
    a = p.parse_args()
    assert re.fullmatch('[0-9a-f]{40}', a.commit)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == a.commit
    publication = json.loads((SUPPORT/'publication.json').read_text())
    assert publication['commit'] == a.commit and publication['remote_ref_verified']
    assert publication['key_file_bytes_verified'] and publication['server_snapshot_mapping_verified']
    summary = json.loads(git_bytes(a.commit, 'manifests/grid-formal-20261015/EXECUTION_SUMMARY.json'))
    assert summary['status'] in ('READY_FOR_REVIEW', 'PARTIAL', 'BLOCKED')
    members = {}

    def add(name, data):
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts and path.suffix not in FORBIDDEN, name
        assert name not in members, name
        assert len(data) < 64*1024**2, name
        if path.suffix in ('.py', '.json', '.jsonl', '.yaml', '.yml', '.md', '.csv', '.log', '.patch'):
            assert not SECRET.search(data), name
        members[name] = data

    def file(name, source):
        assert source.is_file() and not source.is_symlink(), source
        add(name, source.read_bytes())

    names = subprocess.check_output(['git', 'diff', '--name-only', '--diff-filter=ACMRT', BASE, a.commit], cwd=REPO, text=True).splitlines()
    for name in names:
        add('repository/'+name, git_bytes(a.commit, name))
    # Include every computational Python file, even unchanged inherited source.
    source_dirs = ['GRID baseline/src', 'references/wmh-original/GRID/src',
        'experiments/beauty_extensions', 'experiments/preflight_20261015', 'experiments/formal_20261015']
    tracked = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', a.commit, '--', *source_dirs], cwd=REPO, text=True).splitlines()
    for name in tracked:
        if Path(name).suffix == '.py':
            add('fixed-source/'+name, git_bytes(a.commit, name))
    evidence = []
    for job in summary['jobs']:
        assert job['status'] in ('SUCCEEDED', 'FAILED', 'TIMED_OUT', 'CANCELLED'), job['job_id']
        assert job['fetched'] and job['cleanup_verified'], job['job_id']
        # A selective archive intentionally excludes model/data payloads. Never
        # pretend such a directory is a full runner export.
        directory = Path(job.get('selective_export_local') or job['full_export_local'])
        assert directory.is_dir() and not directory.is_symlink()
        status = json.loads((directory/'status.json').read_text())
        assert status['status'] == job['status'] and status['cleanup']['verified']
        for f in sorted(directory.rglob('*')):
            assert not f.is_symlink(), f
            if f.is_file() and f.suffix not in FORBIDDEN and 'checkpoints' not in f.relative_to(directory).parts:
                file('evidence/'+job['job_id']+'/'+str(f.relative_to(directory)), f)
        evidence.append({'job_id': job['job_id'], 'local_evidence': str(directory),
            'collection_kind': 'selective-no-weights' if job.get('selective_export_local') else 'full-export'})
    for job in ('beauty-a-metadata-r3', 'beauty-b-smoke-r3'):
        for name in ('fixed_attention_samples.json', 'test_target_index.jsonl'):
            file('inherited-cohorts/'+job+'/'+name, JOBS/job/'retrieved/output'/name)
    for name in ('AGENTS.md', 'PROJECT_RULES.md', 'REMOTE_EXECUTION.md'):
        file('applied-rules/'+name, Path('/home/addaswsw/recommend')/name)
    # Explicit small support roots only; no temporary tar/ZIP, weights, data or
    # unrelated archive traversal. Raw audit files retain their original bytes.
    for f in sorted(SUPPORT.iterdir()):
        if f.is_file() and f.suffix in ('.json', '.py', '.patch', '.md'):
            file('external-receipts/'+f.name, f)
    for name in ('observations', 'attention-admission', 'mlp-admission', 'results-review-r4',
                 'mlp-results-review', 'attention-results-review', 'resource-hold-attempt-1'):
        root = SUPPORT/name
        if root.is_dir():
            for f in sorted(root.rglob('*')):
                assert not f.is_symlink(), f
                if f.is_file() and f.suffix in ('.json', '.py', '.md', '.yaml', '.csv', '.log'):
                    file('external-receipts/'+str(f.relative_to(SUPPORT)), f)
    readme = f'''# GRID 正式阶段统一审核包

固定 GitHub 提交：{a.commit}
本轮状态：{summary['status']}
入口：https://github.com/woobowen/recommend/blob/{a.commit}/manifests/grid-formal-20261015/FINAL_RECEIPT.md

先读 repository/manifests/grid-formal-20261015/FINAL_RECEIPT.md 与 EXECUTION_SUMMARY.json。
repository/ 保存本轮相对已审核预跑 {BASE} 的实际 Git 字节；fixed-source/ 保存完整计算 Python 源码，方便沿模型/数据/生成/evaluator 阅读。
evidence/ 保存全部本轮终态作业原始 request/status/log、预测、逐样本复算、配置、训练 CSV、attention bundle/NPZ/PNG 等必要轻量证据。训练权重不入包，服务器精确路径/哈希和严格恢复证据见各训练 RESULT 与请求。
完整 Beauty 评测、独立 scratch MLP 训练、固定32 attention 是不同阶段；不能将工程预跑指标当正式结果。失败与修正链原样保留。
inherited-cohorts/ 是预先固定的 Beauty A/B 32样本与全测试目标索引，不重新挑图；Toys/Sports cohort 与索引在对应 metadata 作业中。
每个 attention 的四行是四个自回归 SID query，横轴是 encoder 输入位置；A/B 是历史 SID/SEP，MLP 是聚合物品 token。矩阵为 cross-attention，最后 decoder 层，图对 head 求均值而原始最后层全部 head 保留。actual generation final Top-1 ancestry 与 local first path 分开；看起来相近不证明顺序不变性。
external-receipts/ 保存真实独立审核、资源观测、发布/收集/配置例外证据。proposed-resource-sharing-policy.patch 未应用，不代表新授权或有效执行源码。
applied-rules/ 是显式读取的项目 v1.1 规范。无模型/数据载荷、私钥或旧整套档案。
MANIFEST.json 列出每一成员大小/SHA-256，ZIP SHA-256 在包外；NPZ 请使用 allow_pickle=False。
'''
    add('README.md', readme.encode())
    records = [{'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        for name, data in sorted(members.items())]
    total = sum(r['bytes'] for r in records)
    assert total < 512*1024**2, total
    add('MANIFEST.json', (json.dumps({'commit': a.commit, 'base_commit': BASE,
        'status': summary['status'], 'files': records, 'job_evidence': evidence,
        'uncompressed_bytes_without_manifest': total, 'large_assets_excluded': True,
        'secret_pattern_scan': 'PASS', 'zip_sha256_outside': True}, ensure_ascii=False, indent=2)+'\n').encode())
    destination = SUPPORT/'deliveries'
    destination.mkdir(exist_ok=True)
    archive = destination/'grid-formal-20261015.zip'
    assert not archive.exists(), 'Preserve an existing formal review ZIP; never silently overwrite'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, data in sorted(members.items()):
            z.writestr(name, data)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(members)
        manifest = json.loads(z.read('MANIFEST.json'))
        assert set(z.namelist()) == {r['path'] for r in manifest['files']} | {'MANIFEST.json'}
        for r in manifest['files']:
            data = z.read(r['path'])
            assert len(data) == r['bytes'] and hashlib.sha256(data).hexdigest() == r['sha256'], r['path']
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n')
    unc = subprocess.check_output(['wslpath', '-w', str(archive)], text=True).strip()
    receipt = {'path': str(archive), 'windows_unc': unc, 'bytes': archive.stat().st_size,
        'sha256': digest, 'members': len(members), 'uncompressed_bytes_without_manifest': total,
        'crc_and_each_member_sha256_verified': True, 'secret_pattern_scan': 'PASS', 'commit': a.commit}
    (destination/'zip-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
