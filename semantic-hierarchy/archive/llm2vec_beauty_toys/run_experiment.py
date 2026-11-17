"""Persistent coordinator: GPU text encoding then CPU statistics, per dataset."""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
import argparse
from pathlib import Path
import subprocess
import sys
import time
from runtime_common import read_json, status, write_json, sha
from semantic_metrics import holm_adjust


def finalize(run, cfg):
    summaries = {}
    for dataset in ['beauty','toys']:
        summaries[dataset] = {variant: read_json(run / dataset / variant / 'evaluation/summary.json')
                              for variant in cfg['variants']}
    primary_p = {dataset: summaries[dataset]['title']['main_p_value'] for dataset in summaries}
    adjusted = holm_adjust(primary_p)
    for dataset in summaries:
        summaries[dataset]['title']['main_p_value_holm_two_datasets'] = adjusted[dataset]
    write_json(run / 'combined_results.json', {'datasets':summaries,
                                              'primary_tests_holm_adjusted':adjusted})
    lines = ['# LLM2Vec 对 Qwen SID 的语义细分评估', '',
             '主要证据为第一层到第二层；第三层仅评价至少两件商品的集合并报告覆盖率。',
             '未执行门槛敏感性实验。主输入为标题；标题加品牌为补充对照。', '',
             '| 数据集/输入 | 1→2 平均增量 | 1→2 正增量比例 | 2→3 平均增量 | 完整路径递增比例 | 第三层覆盖率 |',
             '|---|---:|---:|---:|---:|---:|']
    for dataset, variants in summaries.items():
        for variant, summary in variants.items():
            x=summary['observed'];c=summary['coverage']['3']['coverage']
            lines.append(f"| {dataset}/{variant} | {x['delta_12']:.6f} | {x['positive_12']:.2%} | {x['delta_23']:.6f} | {x['path_monotonic']:.2%} | {c:.2%} |")
    lines.extend(['', '1→2 主检验在两数据集之间使用 Holm 校正：', ''])
    for dataset, p in adjusted.items():
        lines.append(f'- {dataset}: p={p:.4g}。')
    lines.extend(['', '随机对照在相同父集合内部保持子节点大小重新划分，999 次。',
                  '95% 区间按第一层子树为簇 bootstrap 2000 次；子树和路径不作为独立商品对处理。',
                  '完整路径主分母要求集合逐步缩小，判定条件为 S1<S2≤S3。',
                  '详细节点、边、全部路径、Top-20 路径、分支区分度和随机分布在各输入的 evaluation/ 下。',
                  '结果验证的是当前独立编码空间中的语义细化，不直接等同于人类类别树。'])
    (run / '实验结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    run=Path(args.run);cfg=read_json(run/'analysis_config.json');launch=read_json(run/'launch.json')
    source=Path(__file__).resolve().parent
    for name,digest in launch['source_sha256'].items():
        assert sha(source/name)==digest, 'Frozen experiment source changed'
    state=run/'experiment_status.json';jobs={};logs=[]
    try:
        status(state,'encoding',run=str(run),gpu_allocation=launch['gpu_allocation'])
        for dataset, devices in launch['gpu_allocation'].items():
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,devices)))
            log=(run/f'{dataset}.encoding.log').open('ab');logs.append(log)
            command=[sys.executable,'-u',str(source/'encoder_worker.py'),'--run',str(run),'--dataset',dataset]
            process=subprocess.Popen(command,env=env,cwd=source,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            jobs[dataset]={'encoding':process,'evaluation':None}
        write_json(run/'processes.json',{d:{'encoding_pid':j['encoding'].pid} for d,j in jobs.items()})
        while True:
            for dataset,job in jobs.items():
                code=job['encoding'].poll()
                if code is not None and code!=0:
                    raise RuntimeError(f'{dataset} encoding failed with exit {code}; inspect encoding log')
                if code==0 and job['evaluation'] is None:
                    log=(run/f'{dataset}.evaluation.log').open('ab');logs.append(log)
                    command=[sys.executable,'-u',str(source/'evaluate_worker.py'),'--run',str(run),'--dataset',dataset]
                    job['evaluation']=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),cwd=source,
                                                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    status(state,'encoding_or_evaluating',dataset_started_evaluation=dataset)
                if job['evaluation'] is not None:
                    result=job['evaluation'].poll()
                    if result is not None and result!=0:
                        raise RuntimeError(f'{dataset} evaluation failed with exit {result}; inspect evaluation log')
            if all(job['evaluation'] is not None and job['evaluation'].poll()==0 for job in jobs.values()):
                break
            time.sleep(5)
        finalize(run,cfg)
        status(state,'complete',finished_unix=time.time(),report=str(run/'实验结果.md'))
    except Exception as exc:
        status(state,'failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        for log in logs:log.close()


if __name__=='__main__':main()
