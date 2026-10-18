#!/usr/bin/env python3
"""Copy a bounded scientific release to wbw; git integration stays with root."""
from pathlib import Path
import hashlib
import json
import sys

BASE = Path(__file__).resolve().parent
REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
TARGET = REPO/'manifests/grid-a2-fd-recovery-20261018/final-results'
STAGE = BASE/'deliveries/final-evidence-staging-v2'
sys.path.insert(0, str(BASE))
from package_delivery import scan_sensitive_bytes, safe_name

def main():
    assert not TARGET.exists()
    TARGET.mkdir(parents=True)
    records = []
    def copy(source, relative):
        safe_name(relative); data = source.read_bytes()
        assert source.is_file() and not source.is_symlink() and len(data) < 4_000_000
        scan_sensitive_bytes(data, relative)
        path = TARGET/relative; path.parent.mkdir(parents=True, exist_ok=True)
        assert not path.exists(); path.write_bytes(data)
        records.append(dict(path=relative, source=str(source), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    for path in sorted((BASE/'final-results').iterdir()):
        if path.is_file():copy(path, path.name)
    for label in ['a2_training','a2_outputs','asset_inventory','delivery_tools']:
        for path in sorted((BASE/'reviews'/label).iterdir()):
            if path.is_file() and path.suffix in ['.json','.md','.py','.sha256']:
                copy(path, 'non-author/'+label+'/'+path.name)
    for name in ['quantization-result-review.json','formal-startup-review.json','quantizer-cpu-products-independent.json']:
        copy(BASE/'reviews/fd'/name, 'non-author/fd/'+name)
    for name in ['cross-domain-evidence-index-with-a2.json','cross-domain-original-metrics-with-a2.json',
                 'cross-domain-original-metrics-with-a2.csv','ORIGINAL_METRICS_WITH_A2.md']:
        copy(BASE/'reviews/cross_domain'/name, 'cross-domain/'+name)
    for path in sorted((BASE/'primary-result-acceptance').iterdir()):
        if path.is_file():copy(path, 'primary/'+path.name)
    copy(BASE/'final-scene/scene-20261018T043428336611Z.json', 'terminal-scene.json')
    copy(BASE/'source-deployment.json', 'source-deployment.json')
    copy(BASE/'queue-deployment.json', 'queue-deployment.json')
    for job in sorted((STAGE/'raw/a2').iterdir()):
        if not job.is_dir() or not job.name.startswith('ga-20261018-sportsa2-'):continue
        for name in ['request.json','accepted.json','status.json','cleanup.json']:
            path=job/name
            if path.is_file():copy(path, 'raw-result-receipts/'+job.name+'/'+name)
        for name in ['RESULT.json','effective_config.yaml','used_config.json','loading.json','checkpoint_protocol.json',
                     'stage-entry.json','stage-exit.json','invocation.json','training_result.json','validations.jsonl']:
            path=job/'output'/name
            if path.is_file():copy(path, 'raw-result-receipts/'+job.name+'/output/'+name)
    for name in ['primary_attention_acceptance.py','primary_a_route_acceptance.py','primary_training_acceptance.py',
                 'capture_final_scene.py','collect_controller_terminal.py','append_final_inputs.py',
                 'assemble_result_report.py','prepare_result_publication.py','package_delivery.py']:
        copy(BASE/name, 'local-execution-tools/'+name)
    manifest = dict(status='SCIENTIFIC_RELEASE_COPIES_VERIFIED_NOT_YET_GIT_PUBLICATION',
                    scientific_execution_commit='85b77f5e2a5b497fdc1f620649ba488fc783a245',
                    source_snapshot_sha256='22286966cd544773f21b96e4185d56c66afd7183fdc4e8888716d8ea1d77acb2',
                    actual_publication_commit_not_inferred=True, files=records)
    (TARGET/'copy-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    readme='''# Sports A2 FD恢复最终科研结果

首先阅读 [FINAL_RESULTS.md](FINAL_RESULTS.md)。四条A路线12条件及新A2全部原计划科研计算/真实非作者审核闭合；旧Beauty/B只继承已固定来源。该目录和晚发表的交付回执不改变实际服务器执行源码85b77f5e2a5b497fdc1f620649ba488fc783a245 / snapshot22286966cd544773f21b96e4185d56c66afd7183fdc4e8888716d8ea1d77acb2。

- 原指标、逐用户配对、全部两路径描述统计：`original-metrics.csv`、`paired-changes.csv`、`attention-means.csv`。
- 8路线24条件完整固定来源：`cross-domain/ORIGINAL_METRICS_WITH_A2.md`及新`*-with-a2.*`，旧pending文件保持历史。
- 实际非作者原件：`non-author/`；新11阶段实际请求/状态/有效配置/结果：`raw-result-receipts/`；45已接收作业和10未提交旧模板、预算/源码/清理：`terminal-scene.json`。
- 大资产路径、大小、原SHA及环境缺口：`non-author/asset_inventory/assets-not-copied.json`。完整原始预测/NPZ/PNG和全部在线观察收入一次最终审核ZIP，不把大型checkpoint、TFRecord或完整环境提交Git。
- `copy-manifest.json`是复制时点的身份凭证。实际固定发布SHA与回读在发布后产生；`scientific-results.json`是先前汇总时点，remaining delivery gates不回写冒充已经完成。
- 最终交付ZIP的真实大小、成员数、整体SHA、WSL/Windows路径及独立整包审核将保存在同目录`delivery/`。先发布科研原件，再制作唯一ZIP，最后仅新增交付回执；这种顺序避免让ZIP自含自身hash或将晚回执提交冒充执行源码。

GPT_UPLOAD_REQUIRED: YES。上传唯一最终ZIP供GPT网页版实际二次审核。未宣称网页版已审；没有新实验/其他分支同步或清理授权。
'''
    (TARGET/'README.md').write_text(readme)
    print(json.dumps(dict(target=str(TARGET), copied_members=len(records),
                         copied_bytes=sum(r['bytes'] for r in records),
                         Git_or_remote_mutations=False),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
