"""Prepare the publication directory from the observed artifacts and independent reviews."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    for name in ('root','review-launch','review-generation'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    root=args.root
    dest=Path(__file__).resolve().parent
    py=[sys.executable,'-B',str(dest/'summarize.py'),'--evidence',str(root)]
    table=subprocess.check_output(py,text=True)
    stats=json.loads((root/'stage-summary.json').read_text())
    audit=json.loads((root/'verified-stage.json').read_text())
    assert audit['status']=='PASS' and audit['observations']['points']==7
    assert stats['full_group_results']=='PENDING' and stats['active_monitoring_stopped'] is True
    evidence=dest/'evidence'
    evidence.mkdir(exist_ok=False)
    for name in (
        'formal-plan.json','authorization.json','budget-before.json','prelaunch.json','parent-request.json',
        'upload-receipts.json','submission-response.json','observer-continuation.json','observe-t0.py',
        'observer-clock-correction.json','observe-t05-t10.py',
        'first-six.jsonl','first-six-capture.json','derived.rank0.py','derived.rank0.diff',
        'derived.rank0.py.receipt.json','derived.rank0.diff.receipt.json','verified-stage.json',
        'window-complete.json','delivery-check.json','stage-summary.json',
    ):
        shutil.copy2(root/name,evidence/name)
    for name in ('observations','server-startup'):
        shutil.copytree(root/name,evidence/name)
    for directory,origin,required in (
        ('review-launch',args.review_launch,('startup-review.json','independent-ssh-live.json','first-six-review.json',
          'first-six-direct-prefix.json','observation-review.json','REVIEW.md','verify.py','verify_observations.py','verify_package.py')),
        ('review-generation',args.review_generation,('live-first-six-review.json','observer-first-six-review.json',
          'verify_formal_rows.py','verify_seven_points.py','verify_live_ranges.py','read_server_prefix.py',
          'seven-point-review.json','live-byte-range-review.json','review-generation.md')),
    ):
        out=evidence/directory
        out.mkdir()
        for name in required:
            path=origin/name
            assert path.is_file(),(directory,name)
            shutil.copy2(path,out/path.name)
        if directory=='review-generation':
            path=origin/'live-prefix'/'direct-prefix-readback.json'
            shutil.copy2(path,out/path.name)
    budget=['','## 预算、估算和交付前复核','',
            f'七点实际跨度 {stats["observed_span_seconds"]:.3f} 秒。各点同一作业身份稳定，未发现异常退出、OOM、超时或输入守卫失败。T+30 的 rank 报告为 {stats["T30_latest_rank_log_count"]:,} 人。最低可用输出空间 {stats["minimum_disk_free_bytes"]/1e12:.3f} TB。','']
    for key,label in [('formal_gpu','正式 GPU device-seconds'),('preflight_gpu','预跑 GPU device-seconds'),('cpu','CPU worker 秒'),('parent_wall','父监督墙钟秒')]:
        record=stats['T30_budget'][key]
        budget.append(f'- {label}。T+30 已用 {record["used_seconds"]:.6f}，余 {record["remaining_seconds"]:.6f}。')
    eta=stats['later_mode_gpu_hours_estimate']
    budget+=['',
       f'严格装载实测 {stats["strict_load_seconds"]:.2f} 秒。按 T+5 至 T+30 记录增量计得 {stats["stable_users_per_second"]:.3f} 用户/秒，避开模型加载。T+30 时 S2 预计还需 {stats["S2_remaining_gpu_hours_at_T30"]:.2f} 小时。按准备阶段各模式耗时比例、实测 S2 速率折算，S3a/S3b/S4 约 {eta["S3a"]:.2f}/{eta["S3b"]:.2f}/{eta["S4"]:.2f} 小时。整组预计还需 {stats["all_remaining_gpu_hours_at_T30"]:.2f} GPU 小时。单卡争用、输入长度及写盘会改变耗时。每项 CPU 评价另计几十秒至数分钟，配对另计数分钟，最终审核和交付另估 20–45 分钟。','',
       f'另在 {stats["delivery_check_ET"]} 完成交付前只读复核，S2 已有 {stats["delivery_check_complete_users"]:,} 条完整预测，父控制器仍正常推进。 [交付前复核](evidence/delivery-check.json) · [机器可读汇总](evidence/stage-summary.json) · [七点验收](evidence/window-complete.json)','']
    header=(root/'phase-header.md').read_text().rstrip()
    tail=(root/'phase-tail.md').read_text().replace('__CODE_FENCE__',chr(96)*3)
    (dest/'README.md').write_text(header+'\n\n'+table+'\n'+'\n'.join(budget)+tail)


if __name__=='__main__':
    main()
