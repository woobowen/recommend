"""Assemble the formal observation ZIP from fixed sources and selected actual evidence."""
import argparse
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from lc_rec.shuffle_v2 import g3_assets as ga

GPU = '86660be5a62c8325caf74cf2810581850ca12803'
CONTROL = 'cb3b1f36d69ef0c1a7f554ef5455026523635f78'
CLOSURE = 'lc_rec/a0_beauty/grid_sid_preparation/g3_conversion/production_preparation/cpu_closure'
PHASE = 'lc_rec/shuffle/results/v2-a0-beauty-g3/launch-observation-r0'
MODULES = ('g3_assets.py','g3_formal.py','g3_control.py','g3_evaluate.py','g3_generate.py','g3_export.py',
           'core.py','reference.py','beauty.py','beauty_control.py','collect.py','control.py',
           'statistics.py','evaluate.py','replay.py','runtime.py','a0_beauty_g3.json')


def export(ref, paths, to):
    archive=subprocess.check_output(['git','archive',ref,*paths])
    to.mkdir(parents=True,exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        assert all(m.isdir() or m.isfile() for m in tar.getmembers())
        tar.extractall(to,filter='data')


def prepare(root, destination):
    destination.mkdir(parents=True,exist_ok=False)
    paths=[f'lc_rec/shuffle_v2/{n}' for n in MODULES]
    paths+=['lc_rec/shuffle/core.py','lc_rec/letter/scripts/formal_evaluate.py','lc_rec/a0_beauty/g3_identity.py',
            'lc_rec/a0_beauty/uneven_sampler.py','lc_rec/a0_beauty/offline_metrics.py',
            'lc_rec/a0_beauty/config/preflight_g3_users_seed42.json',
            'lc_rec/letter/review/final_audit_20261113/external/official_cce.py',
            'lc_rec/letter/review/final_audit_20261113/statistics_verify.py',
            *[f'{CLOSURE}/{n}' for n in ('predictions.py','common.py','closure_config.json')]]
    export(CONTROL,paths,destination/'code')
    original=destination/'gpu-source'
    original.mkdir()
    gpu_deployment=ga.read(root.parent/f'{GPU}.deployment.json')
    gpu_paths={item['path'] for item in gpu_deployment['files']}
    for name in MODULES:
        path=f'lc_rec/shuffle_v2/{name}'
        if path not in gpu_paths:
            continue
        blob=subprocess.check_output(['git','show',f'{GPU}:{path}'])
        if blob!=(destination/'code/lc_rec/shuffle_v2'/name).read_bytes():
            (original/name).write_bytes(blob)
    machinery=destination/'execution-source'
    machinery.mkdir()
    prior=root.parent
    for name in ('runner.py','runner.deployment.json',f'{GPU}.deployment.json',f'{CONTROL}.deployment.json'):
        shutil.copy2(prior/name,machinery/name)
    for name in ('transport.py','gridctl.py'):
        shutil.copy2(prior.parent/'sid-extensions-p0/scripts'/name,machinery/name)
    for pin in (GPU,CONTROL):
        manifest=ga.read(machinery/f'{pin}.deployment.json')
        content={Path(x['path']).name:x for x in manifest['files'] if x['path'].startswith('lc_rec/shuffle_v2/')}
        for name in MODULES:
            if name not in content: continue
            path=(original/name if pin==GPU and (original/name).is_file()
                  else destination/'code/lc_rec/shuffle_v2'/name)
            assert ga.sha(path)==content[name]['sha256'],(pin,name)
    frozen=destination/'frozen-input'
    shutil.copytree(root/'frozen-input',frozen)
    cfg=ga.read(ga.CONFIG)
    assert ga.sha(frozen/'normal-rank0.json')==cfg['normal_rank_sha256']
    assert ga.sha(frozen/'cpu/inputs.jsonl')==ga.read(frozen/'cpu/audit.json')['manifest_sha256']


def finish(root, destination, publication):
    receipt=ga.read(publication)
    commit=receipt['published_commit']
    temp=root/'published-export'
    export(commit,[PHASE],temp)
    source=temp/PHASE
    shutil.copytree(source,destination/'phase')
    shutil.copy2(publication,destination/'publication.json')
    command=('PYTHONPATH=code python3 -B phase/verify_stage.py '
             '--evidence phase/evidence --data-root frozen-input/data '
             '--cpu-audit frozen-input/cpu/audit.json --normal-rank frozen-input/normal-rank0.json')
    names=sorted(p.name for p in (destination/'phase/evidence/review-launch').iterdir())
    assert 'startup-review.json' in names and 'first-six-review.json' in names
    ticks=chr(96)*3
    readme=[
        '# 第三组 G3 Beauty Shuffle V2 正式启动观察审核包','',
        '正式父控制器已在服务器独立接管。同一 S2 作业的七点观察跨度达到 30 分钟，完整组结果仍待生成、评价和配对。此包保存当时固定字节边界的证据。','',
        '- [启动与观察报告](phase/README.md)。方案、时间、预算、估算、任务编号和接续查询。',
        '- [正式计划](phase/evidence/formal-plan.json)、[本次授权](phase/evidence/authorization.json)、[父请求](phase/evidence/parent-request.json)、[服务器回读](phase/evidence/server-startup/readback.json)。父与 S2 的接受和实际载荷保存在相邻目录。',
        '- [首六条原始 JSONL](phase/evidence/first-six.jsonl)、[截取规则与实际 rank](phase/evidence/first-six-capture.json)、[实际派生源码](phase/evidence/derived.rank0.py)。完整 CPU manifest 位于 frozen-input/cpu，原采样、Normal 模型身份和 Beauty 固定数据都随包提供。',
        '- [非作者启动核验](phase/evidence/review-launch/startup-review.json)、[首六条独立边界检查](phase/evidence/review-launch/first-six-review.json)、[主控离线验证](phase/evidence/verified-stage.json)。非作者直接读取服务器并核对保存快照。',
        '- [GitHub 发布回读](publication.json)。code 保存固定 CPU/控制器版本和离线依赖，gpu-source 保存有字节差异的 GPU 版本文件，execution-source 保存实际 runner、传输客户端和两个部署清单。冻结原生成入口和原 loader 位于 frozen-input/frozen-source。',
        '- [交付前复核](phase/evidence/delivery-check.json) 独立于七点。服务器输出在继续增长，包中记录均注明观测时点与范围。','',
        '## 实际解压后复核','',
        '在解压根目录用 Bash、Python 3.10+ 和 NumPy 运行。复核从保存的模型证据开始，不调用模型或服务器。','',
        ticks+'bash',command,ticks,'',
        '原冻结 Normal 全量、模型权重和前置审核包继续保存在既有位置。本包收录启动观察所需的身份回执、CPU 输入绑定与边界证据。','',
    ]
    (destination/'README.md').write_text('\n'.join(readme))


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action',choices=('prepare','finish'))
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--destination',required=True,type=Path)
    p.add_argument('--publication',type=Path)
    a=p.parse_args()
    if a.action=='prepare':
        prepare(a.root,a.destination)
    else:
        assert a.publication
        finish(a.root,a.destination,a.publication)


if __name__=='__main__':
    main()
