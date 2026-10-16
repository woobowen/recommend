"""Bounded review ZIP from published Git bytes and verified original task evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

REPO=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT=Path('/home/addaswsw/recommend-support/grid-shuffle-20261017')
JOBS=Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server')
PREFIX='manifests/grid-shuffle-20261017/'
SECRET=re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{30,}')
TEXT={'.py','.json','.jsonl','.md','.yaml','.yml','.csv','.log','.txt','.patch'}
ALLOWED=TEXT|{'.npz','.png','.gz'}


def sha(value):return hashlib.sha256(value).hexdigest()
def read(path):return json.loads(path.read_bytes())
def git_bytes(commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=REPO)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--commit',required=True)
    args=parser.parse_args();assert re.fullmatch('[0-9a-f]{40}',args.commit)
    publication=read(SUPPORT/'publication-content.json')
    assert publication['commit']==args.commit and publication['remote_ref_verified'] and publication['key_file_bytes_verified']
    summary=json.loads(git_bytes(args.commit,PREFIX+'EXECUTION_SUMMARY.json'))
    assert summary['status']=='PARTIAL' and summary['qualified_B_conditions_complete']
    assert summary['active_job_ids']==[]
    content_review=read(SUPPORT/'delivery-audit/content-independent-review.json')
    assert content_review['status']=='CONTENT_ACCEPTANCE_PASS' and content_review['content_commit']==args.commit
    assert content_review['summary']['sha256']==sha(git_bytes(args.commit,PREFIX+'EXECUTION_SUMMARY.json'))
    assert content_review['publication']['sha256']==sha((SUPPORT/'publication-content.json').read_bytes())
    statistics=read(SUPPORT/'attention-statistics/attention-descriptive-statistics.json')
    assert (SUPPORT/'attention-statistics/attention-descriptive-statistics.json').read_bytes()==git_bytes(
        args.commit,PREFIX+'attention-statistics/attention-descriptive-statistics.json')
    selected={}
    for row in statistics['jobs']:
        paths={'output/batch-00/figures/sample_000_cross_attention.png'}
        for values in row['least_similar_three_under_declared_mean_cosine'].values():
            paths.update(value['plot_relative'] for value in values)
        selected[row['job_id']]=paths
    members={}

    def add(name,data):
        p=Path(name)
        assert not p.is_absolute() and '..' not in p.parts and '\\' not in name
        assert name not in members and p.suffix in ALLOWED,name
        if p.suffix in TEXT:assert not SECRET.search(data),name
        members[name]=data

    def file(name,path):
        assert path.is_file() and not path.is_symlink(),path
        add(name,path.read_bytes())

    tracked=subprocess.check_output(['git','ls-tree','-r','--name-only',args.commit,'--',
        'experiments/shuffle_20261017',PREFIX],cwd=REPO,text=True).splitlines()
    for name in tracked:add('repository/'+name,git_bytes(args.commit,name))
    source_dirs=['GRID baseline/src','references/wmh-original/GRID/src','experiments/beauty_extensions',
        'experiments/preflight_20261015','experiments/formal_20261015']
    source_files=subprocess.check_output(['git','ls-tree','-r','--name-only',args.commit,'--',*source_dirs],cwd=REPO,text=True).splitlines()
    for name in source_files:
        if Path(name).suffix=='.py':add('fixed-source/'+name,git_bytes(args.commit,name))
    for name in ['attention_statistics.py','attention-results-review/verify_attention_statistics.py',
                 'results-review/beauty_paired_shuffle_analysis.py']:
        add('fixed-source/manifests/grid-formal-20261015/'+name,
            git_bytes(args.commit,'manifests/grid-formal-20261015/'+name))
    for route in ('a','b'):
        for n in (0,3,4):
            name=f'experiments/formal_20261015/specs/beauty-{route}-{n}.json'
            add('inherited-beauty/'+name,git_bytes(summary['inherited_Beauty_anchor'],name))
    for name in ['results-review/beauty-six-formal-results-independent-summary.json',
                 'results-review/beauty-paired-shuffle-analysis.json',
                 'attention-statistics/attention-descriptive-statistics.json']:
        add('inherited-beauty/'+name,git_bytes(summary['inherited_Beauty_anchor'],'manifests/grid-formal-20261015/'+name))
    raw_receipts=[]
    for job in summary['jobs']:
        root=Path(job['local_evidence']);manifest=read(root/'export-manifest.json')
        request=read(root/'request.json');status=read(root/'status.json')
        assert request['job_id']==status['job_id']==job['job_id']
        assert status['status']==job['status'] and status['exit_code']==job['exit_code']
        assert status['cleanup']['verified']
        listed={r['path']:r for r in manifest['files']};assert len(listed)==len(manifest['files'])
        actual={str(f.relative_to(root)) for f in root.rglob('*') if f.is_file()}
        assert actual==set(listed)|{'export-manifest.json'}
        image_index=[];checked=0;png_added=0
        available_png={name for name in listed if name.endswith('.png')}
        if job['job_id'] in selected:assert selected[job['job_id']]<=available_png
        for name,row in listed.items():
            p=Path(name);assert not p.is_absolute() and '..' not in p.parts
            value=(root/name).read_bytes()
            assert len(value)==row['bytes'] and sha(value)==row['sha256'],name
            checked+=1
            if p.suffix=='.png':
                image_index.append(row)
                if name not in selected.get(job['job_id'],set()):continue
                png_added+=1
            if p.suffix in ALLOWED:add('evidence/'+job['job_id']+'/'+name,value)
        file('evidence/'+job['job_id']+'/export-manifest.json',root/'export-manifest.json')
        if image_index:
            add('evidence/'+job['job_id']+'/ALL_PNG_IDENTITIES.json',(json.dumps(image_index,indent=2)+'\n').encode())
        raw_receipts.append({'job_id':job['job_id'],'export_files_verified':checked,'all_original_PNG':len(image_index),
            'PNG_in_review_ZIP':png_added,'original_manifest_sha256':sha((root/'export-manifest.json').read_bytes())})
    assert len(selected)==6
    assert sum(name.startswith('evidence/') and name.endswith('/bundle/arrays.npz') for name in members)==24
    assert sum(name.startswith('evidence/') and name.endswith('/bundle/record.json') for name in members)==24
    allowed_dirs=['asset-audit','source-audit','results-audit','delivery-audit','attention-statistics','observations','approvals']
    for directory in allowed_dirs:
        for p in sorted((SUPPORT/directory).rglob('*')):
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix in (TEXT|{'.gz'}):
                if p.name.endswith(('.partial','.lock')):continue
                file('external-receipts/'+str(p.relative_to(SUPPORT)),p)
    for name in ['takeover.json','INSTALLATION_LOG.json','presentation-selection.json','correction-round-1.json',
                 'correction-round-2.json','correction-round-3.json','publication-content.json','final-state-query.json']:
        file('external-receipts/'+name,SUPPORT/name)
    for name in ['AGENTS.md','PROJECT_RULES.md','REMOTE_EXECUTION.md']:
        file('applied-rules/'+name,Path('/home/addaswsw/recommend')/name)
    text=f'''# Toys / Sports Shuffle 正式审核包

内容固定提交：{args.commit}
GitHub入口：https://github.com/woobowen/recommend/blob/{args.commit}/{PREFIX}FINAL_REPORT.md
总体状态：PARTIAL。两域 B 的六项全量评测、六组 fixed32 及2304条路径/行对统计已闭合；Toys A选择与上游来源未闭合，Sports A缺完整匹配链。Beauty继承 {summary['inherited_Beauty_anchor']}，本轮未重跑。

先读 repository/{PREFIX}FINAL_REPORT.md、EXECUTION_SUMMARY.json 和外部非作者审核。
evidence/ 保留所有本轮正式及失败作业的原request/status/log/配置/预测/逐用户复算、24份完整NPZ与record.json、beam/heads/mask/实际排列、OFF/ON和plot receipts。
每条件32张PNG完整保留在服务器和已验证本地export；审核ZIP只展示事先固定的首样本与两条路径分别最不相似三样本，全部PNG身份另列。没有重绘、逐行归一化或删样本。
actual final Top1 ancestry 与 local first parent 是两条独立路径。六heads均值图的四行是四个autoregressive query步骤；横轴是历史SID/SEP输入位置。TV/JS归一化只发生在描述统计，不修改原矩阵或PNG。
fixed-source/ 是固定Git计算源码，repository/ 是新任务代码/配置/汇总/统计/审查材料。原资产、checkpoint与全量TFRecord留服务器，路径/hash/恢复入口见报告与manifest。
external-receipts/ 记录真实非作者审查、选模原件、失败和修正链、资源与发布证据。附加CPU callback v1失败原记录未隐藏，不能称该附加补证已成功。
MANIFEST.json列出全部其他成员字节数与SHA。包外intended-members.json列全部成员，包括MANIFEST；ZIP整体SHA和最终容器审核置包外并在最终GitHub交付回执发布，避免自引用。NPZ用allow_pickle=False读取。
本包用于GPT网页版二次审核；内部通过不等于科学假设成立，也不等于网页版已经审核。单seed42，不跨未对齐Shuffle列作因果比较，不把A/B绝对指标差归因于Shuffle。
'''
    add('README.md',text.encode())
    records=[{'path':name,'bytes':len(data),'sha256':sha(data)} for name,data in sorted(members.items())]
    add('MANIFEST.json',(json.dumps({'commit':args.commit,'status':summary['status'],'files':records,'raw_evidence':raw_receipts},indent=2)+'\n').encode())
    intended={'files':[{'path':name,'bytes':len(data),'sha256':sha(data)} for name,data in sorted(members.items())]}
    archive=SUPPORT/f'GRID_Toys_Sports_Shuffle_GPT_Review_{args.commit[:12]}.zip'
    assert not archive.exists()
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(members.items()):z.writestr(name,data)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(members)
        for row in intended['files']:
            value=z.read(row['path']);assert len(value)==row['bytes'] and sha(value)==row['sha256']
    data=archive.read_bytes();digest=sha(data)
    assert len(data)<128*1024**2,'Review ZIP unexpectedly large; preserve it and inspect the manifest'
    archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n')
    intended_path=SUPPORT/'intended-members.json';intended_path.write_text(json.dumps(intended,indent=2)+'\n')
    receipt={'status':'ZIP_BUILT_SELF_VERIFIED_PENDING_NONAUTHOR_CONTAINER_REVIEW','content_commit':args.commit,
        'path':str(archive),'windows_path':subprocess.check_output(['wslpath','-w',str(archive)],text=True).strip(),
        'bytes':len(data),'sha256':digest,'member_count':len(members),'uncompressed_bytes':sum(map(len,members.values())),
        'CRC_all_pass':True,'member_hashes_all_pass':True,'no_duplicate_or_dangerous_paths':True,'secret_pattern_scan_pass':True,
        'NPZ_members':sum(name.endswith('.npz') for name in members),'selected_PNG_members':sum(name.endswith('.png') for name in members),
        'intended_members':str(intended_path),'intended_members_sha256':sha(intended_path.read_bytes())}
    (SUPPORT/'package-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
