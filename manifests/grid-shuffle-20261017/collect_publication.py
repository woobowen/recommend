"""Publishable receipts/summaries and prespecified original-byte attention showcase."""
import hashlib
import json
from pathlib import Path
import shutil

REPO=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT=Path('/home/addaswsw/recommend-support/grid-shuffle-20261017')
JOBS=Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server')
OUT=REPO/'manifests/grid-shuffle-20261017'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def copy(source,target):
    assert source.is_file() and not source.is_symlink()
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target);assert sha(source)==sha(target)


def main():
    paths=[]
    for directory in ('asset-audit','source-audit','results-audit','delivery-audit','attention-statistics'):
        for path in sorted((SUPPORT/directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.json','.md','.py','.yaml','.yml'):
                target=OUT/path.relative_to(SUPPORT);copy(path,target);paths.append(str(target.relative_to(REPO)))
    for name in ('INSTALLATION_LOG.json','presentation-selection.json','final-state-query.json','GitHub-readback-client-check.json',
                 'correction-round-1.json','correction-round-2.json','correction-round-3.json'):
        copy(SUPPORT/name,OUT/name)
    for path in sorted(JOBS.glob('gs-20261017-*')):
        root=path/'retrieved'
        for name in ('request.json','accepted.json','status.json','cleanup.json','export-manifest.json',
            'output/RESULT.json','output/invocation.json','output/loading.json','output/used_config.json',
            'output/effective_config.yaml','output/intervention_trace.json','output/test/recomputed_metrics.json',
            'output/test/metrics.json','output/test/lifecycle_cleanup.json'):
            if (root/name).is_file():copy(root/name,OUT/'runs'/path.name/name)
    stats=json.loads((SUPPORT/'attention-statistics/attention-descriptive-statistics.json').read_text())
    images=[]
    for row in stats['jobs']:
        job=row['job_id'];root=JOBS/job/'retrieved'
        selected={'output/batch-00/figures/sample_000_cross_attention.png'}
        for values in row['least_similar_three_under_declared_mean_cosine'].values():selected.add(values[0]['plot_relative'])
        all_images=[]
        for p in sorted((root/'output').glob('batch-*/figures/*.png')):
            all_images.append({'relative_path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)})
        assert len(all_images)==32
        for name in sorted(selected):copy(root/name,OUT/'attention-showcase'/job/name)
        images.append({'job_id':job,'selection_rule':'Prespecified cohort index0 plus each path least mean pair cosine; original PNG bytes unchanged',
            'selected_relative_paths':sorted(selected),'all_32_original_PNGs':all_images,
            'server_job_root':'/home/bwwu/grid-audits/sid-extensions-p0/jobs/'+job,
            'original_NPZ_files':[{'relative_path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted((root/'output').glob('batch-*/bundle/arrays.npz'))]})
    (OUT/'attention-showcase/IDENTITIES.json').write_text(json.dumps(images,indent=2)+'\n')
    for name in ('shuffle.md','attention.md'):
        p=REPO/'docs/sid-extensions/experiments'/name;text=p.read_text()
        marker='Toys/Sports Shuffle 正式补齐（2026-10-17）'
        if marker not in text:
            text+='\n\n## '+marker+'\n\nToys/Sports B 的 Normal、Shuffle3、Shuffle4 完整测试、对应六组 fixed32 和2304条注意力描述统计已完成并经真实非作者审核；Beauty仅继承。Toys A选择/上游来源未闭合，Sports A缺完整配套链，总状态 PARTIAL。原始预测、24份NPZ和全部192张PNG保留服务器；代码、配套身份、完整指标/配对、统计、失败记录及审核入口见 [正式报告](../../../manifests/grid-shuffle-20261017/FINAL_REPORT.md)。单seed42，不将不同A/B基础条件或图像相似解释为严格置换不变。\n'
            p.write_text(text)
    print(json.dumps({'copied_support_records':len(paths),'attention_conditions':len(images),'source_assets_changed':False}))


if __name__=='__main__':main()
