"""Build a small, verified review ZIP after fixed wbw publication."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

from .deploy import SUPPORT
from experiments.beauty_extensions.common import REPO, sha256, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--commit',required=True)
    a=p.parse_args();assert re.fullmatch('[0-9a-f]{40}',a.commit)
    publication=json.loads((SUPPORT/'publication.json').read_text())
    assert publication['commit']==a.commit and publication['remote_ref_verified']
    summary=json.loads(subprocess.check_output(['git','show',a.commit+':manifests/grid-shuffle-mlp-preflight-20261015/EXECUTION_SUMMARY.json'],cwd=REPO))
    status=summary['status']
    assert status in ('PREFLIGHT_READY_FOR_REVIEW','PARTIAL_PREFLIGHT','BLOCKED')
    receipt=json.loads((SUPPORT/'collection-receipt.json').read_text());collected=Path(receipt['directory'])
    members={}
    def add(name,data):
        assert name not in members
        q=Path(name);assert not q.is_absolute() and '..' not in q.parts
        assert q.suffix not in ('.pt','.ckpt','.pkl','.pyc','.tfrecord','.npz')
        members[name]=data
    names=subprocess.check_output(['git','diff','--name-only','6b39c4323ab9e7c3a369aab6dab6eb59d6129e2c',a.commit],cwd=REPO,text=True).splitlines()
    for name in names:add('repository/'+name,subprocess.check_output(['git','show',a.commit+':'+name],cwd=REPO))
    for f in sorted(collected.rglob('*')):
        if f.is_file():assert not f.is_symlink();add('evidence/'+str(f.relative_to(collected)),f.read_bytes())
    external=['publication.json','collection-receipt.json','terminal-states.json','task.json','rounds.json',
        'asset-presence.json','remote-precheck.json','framework-lifecycle-source.txt','installed-checkpoint-source.json',
        'resource-process-check.json','resource-check-after-cpu.json','resource-check-0610.json','resource-final-check.json',
        'b-shuffle-submission-attempt.json','gsp-20261015-b-shuffle-r1-resource-check.json','round1-final-source.json',
        'shared-preflight-authorization.json','shared-resource-baseline.json','network-recovery-check.json']
    for name in external:add('external-receipts/'+name,(SUPPORT/name).read_bytes())
    for f in sorted(SUPPORT.glob('gsp-20261015-*-shared-resource-check.json')):
        add('external-receipts/resource-admission/'+f.name,f.read_bytes())
    for f in sorted(SUPPORT.glob('*-deployment.json')):add('external-receipts/deployments/'+f.name,f.read_bytes())
    for name in ['work.log','request.json','status.json']:add('prior-B-failure/'+name,(SUPPORT/'prior-failure'/name).read_bytes())
    for name in ['AGENTS.md','PROJECT_RULES.md','REMOTE_EXECUTION.md']:
        add('applied-rules/'+name,(Path('/home/addaswsw/recommend')/name).read_bytes())
    instruction=Path('/home/addaswsw/recommend/GRID_Shuffle_AB_MLP_Toys_Sports_Preflight_Codex_20261015.txt')
    add('task-instructions/'+instruction.name,instruction.read_bytes())
    archive_root=Path('/home/addaswsw/recommend-support/wmh-a800-audit-20261010')
    restored=archive_root/'preservation/A800-grid-20261010-v1/restored/A800_GRID'
    inherited=[restored/'logs/inference/runs/2025-09-22/08-44-16/.hydra/config.yaml',
        restored/'logs/inference/runs/2025-09-22/10-05-01/.hydra/config.yaml',
        restored/'logs/train/runs/2025-10-22/21-25-31/.hydra/config.yaml',
        restored/'logs/train/runs/2025-10-22/21-28-23/.hydra/config.yaml']
    for digest in ['fb215c22cdc413fc81233354439801a93eaa2d5b01ca328fdfc71dadb2a9d028',
                   'baf73e828cbee108864e24933abfc76b8475b427f4743defd1ddce62e41df6f2']:
        inherited.append(archive_root/'analysis/a800-baseline/nested-inspection-b00-b05-v1'/(digest+'.json'))
    for f in inherited:
        assert f.is_file() and not f.is_symlink() and f.stat().st_size < 2*1024**2
        add('inherited-source-evidence/'+str(f.relative_to(archive_root)),f.read_bytes())
    readme=f'''# GRID 第一阶段审核包 — {status}

固定 wbw 提交：{a.commit}
入口：https://github.com/woobowen/recommend/blob/{a.commit}/manifests/grid-shuffle-mlp-preflight-20261015/EXECUTION_SUMMARY.json

先读该 summary、shuffle-a-execution-review.json、shuffle-b-review.json、source-conditions-review.json，再读两份实验说明。
本轮终态与逐线准入以固定提交中的 EXECUTION_SUMMARY.json 为准，共 {len(summary['jobs'])} 个已接收 job。
A/B 每条件31条固定test及11条val，Toys/Sports各19条test及11条val；训练限每域2次更新及独立恢复后的1次更新。
用户后续明确允许有足够余量时共享单张GPU；共享授权、连续资源基线及每个实际GPU job的监视记录分别保留。此前BUSY拒绝和CPU检查均为真实历史，不作为当前资源结论。
没有任何全量评价、正式训练或正式attention。旧Beauty交付状态仍PARTIAL_BLOCKED。

repository/ 是固定Git提交的变更字节；evidence/ 是原request/status/log、有效配置、真实预跑结果和逐样本预测（另含collection manifest）。
external-receipts/ 是实际部署/传输、资源观察与授权、框架源与GitHub回读/服务器对应回执。
prior-B-failure/ 是旧beauty-b-smoke-r3原件的核验副本，不能冒充本轮实跑。
inherited-source-evidence/ 是Toys两套历史配置及惰性checkpoint metadata，不能证明缺失的Sports历史或作者Python绑定。
task-instructions/ 和 applied-rules/ 固定本次授权及边界。

逐线工程门禁必须由真实执行及非作者审核支持，不能用静态审核/CPU PASS代替REAL_PREFLIGHT_PASS；未来Toys/Sports正式研究条件另审。
所有候选formal_approved=false；未创建有效正式授权token。未来研究配置/训练预算另审。
模型、SID、原数据和旧Beauty完整结果不放入本包，保持服务器原路径。MANIFEST.json逐项列SHA与大小；ZIP整体SHA仅写在包外。
'''
    add('README.md',readme.encode())
    secret=re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{30,}')
    records=[]
    for name,data in sorted(members.items()):
        assert not secret.search(data),name
        records.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    add('MANIFEST.json',(json.dumps({'commit':a.commit,'status':status,'files':records,
        'secret_pattern_scan':'PASS','model_and_source_data_payloads_excluded':True,'zip_sha_outside':True},ensure_ascii=False,indent=2)+'\n').encode())
    destination=SUPPORT/'deliveries';destination.mkdir(exist_ok=True)
    path=destination/'grid-shuffle-mlp-preflight-20261015.zip'
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(members.items()):z.writestr(name,data)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(members)
        m=json.loads(z.read('MANIFEST.json'))
        assert set(z.namelist())=={r['path'] for r in m['files']}|{'MANIFEST.json'}
        for r in m['files']:
            data=z.read(r['path']);assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
    digest=sha256(path);path.with_suffix('.zip.sha256').write_text(digest+'  '+path.name+'\n')
    result={'path':str(path),'windows_unc':subprocess.check_output(['wslpath','-w',str(path)],text=True).strip(),
        'bytes':path.stat().st_size,'sha256':digest,'members':len(members),'crc_and_member_sha256_verified':True,
        'secret_pattern_scan':'PASS','commit':a.commit,'status':status}
    write_json(destination/'zip-receipt.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
