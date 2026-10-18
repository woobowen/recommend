#!/usr/bin/env python3
"""Append completed scientific evidence without overwriting historical captures."""
from pathlib import Path
import datetime as dt
import hashlib
import json

BASE = Path(__file__).resolve().parent
STAGE = BASE/'deliveries/final-evidence-staging-v2'
EXCLUDED = {'raw', 'terminal_outputs', 'preparation', 'preparation-v2', 'terminal-originals', '__pycache__'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def keep(path):
    assert not path.is_symlink(), str(path)
    return path.is_file() and path.suffix not in ['.pyc', '.ckpt', '.bin', '.safetensors'] and not path.name.endswith('.tar.gz')

def main():
    assert not (STAGE/'assembly/final-inputs.json').exists()
    records = []
    def copy(path, relative):
        assert keep(path) and path.stat().st_size < 100_000_000
        data = path.read_bytes(); target = STAGE/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == data, 'Historical copy remains immutable: '+str(target)
        else:
            target.write_bytes(data)
        assert target.stat().st_ino != path.stat().st_ino
        records.append(dict(source=str(path), path=relative, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    def tree(directory, prefix, skip=()):
        for path in sorted(directory.rglob('*')):
            relative = path.relative_to(directory)
            if any(p in skip or p == '__pycache__' or p.startswith('fixture') for p in relative.parts):
                continue
            if keep(path):
                copy(path, str(Path(prefix)/relative))
    tree(BASE/'final-results', 'results')
    copy(BASE/'final-results/RESULTS.md', 'RESULTS.md')
    tree(BASE/'primary-result-acceptance', 'primary/final')
    tree(BASE/'monitor', 'execution/online-monitoring')
    tree(BASE/'final-scene', 'execution/final-scene')
    controller = BASE/'controller-terminal/20261018T043815346507Z'
    tree(controller, 'execution/final-controller')
    for label in ['fd', 'delivery_tools', 'a2_training', 'a2_outputs', 'asset_inventory', 'cross_domain']:
        tree(BASE/'reviews'/label, 'review/final/'+label, EXCLUDED)
    copy(BASE/'reviews/asset_inventory/assets-not-copied.json', 'assets-not-copied.json')
    for path in sorted(BASE.glob('*.py')):
        copy(path, 'tools/final-operations/'+path.name)
    # Reviewer captures may wrap raw status with query fields. Preserve each
    # unique byte sequence; identical artifacts get explicit in-ZIP aliases.
    identities = {}
    for path in sorted(STAGE.rglob('*')):
        if path.is_file():
            identities.setdefault((path.stat().st_size, sha(path)), []).append(path.relative_to(STAGE).as_posix())
    aliases = []
    for label, directory in [('a2-outputs-terminal', BASE/'reviews/a2_outputs/terminal_outputs/raw'),
                             ('a2-outputs-preparation', BASE/'reviews/a2_outputs/preparation-v2/raw'),
                             ('a2-training-terminal', BASE/'reviews/a2_training/terminal-originals')]:
        for path in sorted(directory.rglob('*')):
            if '__pycache__' in path.parts or not keep(path):
                continue
            key = (path.stat().st_size, sha(path))
            if key not in identities:
                relative = 'review/final/unique-capture-inputs/'+label+'/'+path.relative_to(directory).as_posix()
                copy(path, relative); identities[key] = [relative]
            aliases.append(dict(source=str(path), bytes=key[0], sha256=key[1],
                                canonical_ZIP_member=identities[key][0],
                                all_existing_equal_members=identities[key]))
    alias_path = STAGE/'assembly/reviewer-input-path-map.json'
    assert not alias_path.exists()
    alias_path.write_text(json.dumps(dict(status='EVERY_REVIEWER_CAPTURE_INPUT_MAPPED_TO_EXACT_BYTES',
                  inputs=aliases, synthetic_fixtures_included=False), ensure_ascii=False, indent=2)+'\n')
    readme = '''# GRID A2 专项恢复：最终科研证据

先读 `RESULTS.md`。全部计算与各项科研审核已经闭合；最终固定GitHub发表及整包字节验收见 `publication/` 与包外交付回执。本文只说明实际成员，不把历史PENDING快照当当前科研状态。

## 当前有效入口

- `results/scientific-results.json`、`original-metrics.csv`、`paired-changes.csv`、`attention-means.csv`：四条A路线12组完整原指标、配对与两路径统计。
- `review/final/fd/quantization-result-review.json`：唯一正式恢复的30-step/3层/双GPU、新SID与FD实际证据。
- `review/final/a2_training/final_training_review.json`及`FINAL_TRAINING_REVIEW.md`：7900完整更新、79次全量validation、best6900、实际best/final严格载荷验收。
- `review/final/a2_outputs/A2_OUTPUTS_NONAUTHOR_REVIEW.json`、`A2_OUTPUTS_REVIEW.md`和`a2-png-pixel-consistency.json`：全35598×3原预测、12 NPZ/96 PNG、两路径及1152条统计的独立验收。
- `review/toys/TOYS_INDEPENDENT_REVIEW.md`与`review/sports_a1/REPORT.md`：两Toys候选与Sports A1原始结果真实非作者审核。
- `review/final/cross_domain/cross-domain-evidence-index-with-a2.json`及`cross-domain-original-metrics-with-a2.*`：8路线24条件完整入口。无后缀旧index的PENDING仅为早期记录。
- `review/final/delivery_tools/scientific-results-cross-review.json`与`terminal-scene-cross-review.json`：跨项原始身份、预算、状态、源码和研究边界审核。最终交付审核会另附。
- `review/final-internal-review.json`：固定源码发表和完整staging核验后的科研内部结论；ZIP本身的真实字节/CRC/敏感信息复核在序列化后进行，整体SHA必须在包外。

## 原始证据与时间顺序

- `raw/toys/`：两候选全部原预测、三条件fixed32及统计；`raw/sports_a1/`：A1原训练与三条件产物；`raw/a2/`：本次新11阶段的实际request/status/cleanup/logs/有效配置/产物。没有checkpoint模型载荷、完整TFRecord或T5权重。
- 每attention条件为4批×8人，各批`bundle/record.json`引用同目录`arrays.npz`（逐数组shape/dtype/SHA），`figures/plot_receipt.json`绑定PNG及共同色阶。所有384张原图与48 NPZ均保留，不挑图。
- 数值表均是原始evaluator值，独立float64仅用于交叉校验。原非法十候选保留，没有过滤/补齐/改分/重排。fixed32来自evaluation，正式完整推荐指标来自testing。各条件内的query-pair统计不是跨条件attention距离。
- `primary/final/`：主控独立读原预测、训练记录和注意力数组的验收。`primary/quantization/quantization-numpy-arrays.npz`保留可无需Torch读取的新SID及原量化预测数值。
- `history/original-emfile-review/`：原专项ZIP解包成员全部只读继承。原ZIP整体SHA为3be1733fb8148d8a3018682fc9c67dc7e4b94beb3f5466ab1501d2e736be90c3，原失败-9与EMFILE不改。
- `engineering/`、`execution/compiled/`、`execution/online-monitoring/`、`execution/final-controller/`、`execution/final-scene/`：预提交等价验证、真实新请求与固定身份、在线观察、实际控制器独立终态及预算清理。内层queue的TERMINAL_PENDING_RESULT_REVIEW是计算结束时原件；外层job SUCCEEDED/0/cleanup，最终科研审核另外签署，没有回写历史状态。
- `review/`中早期PENDING、拒绝的未提交c31候选与本地包装工具失败是历史取证；当前结论以`review/final/`及上述入口为准，绝不把合成测试fixture当真实作业。所有fixture/fixture-*、私钥和token均未收集。

## 版本、路径和恢复

- 原科研b44399926c98369191c2ff5c2425377c8c07d3c8、共享调度6dafbc7215b58a2dbb47905a687c930caecf7b44、上次交付9ab3d0ea5d32db00e278f74c599a1943885453db和本次正式执行85b77f5e2a5b497fdc1f620649ba488fc783a245分开保留。新source目录为完整262文件实际Git/snapshot字节；original-science目录仅为注明的选定原文件，不冒充完整旧服务器部署。
- 服务器执行根/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/snapshots/22286966cd544773；完整snapshot SHA22286966cd544773f21b96e4185d56c66afd7183fdc4e8888716d8ea1d77acb2；解释器/data/bwwu/envs/grid-20261009-e61baf3/bin/python。
- `assets-not-copied.json`提供21个模型/SID/T5/量化文件和1293个数据成员的真实远端路径、bytes、原始SHA及只读恢复命令；目录无伪造单一hash，env完整镜像缺口明确。它是本轮事后交付索引，不冒充历史运行已使用的manifest。
- `assembly/*.json`保存每次静态/终态/最终追加复制的逐文件来源；`assembly/reviewer-input-path-map.json`将实际非作者原始输入路径映射到包内字节相同的成员，避免重复携带整套预测。独立capture的query包装status若字节不同，另保留唯一原件。审核脚本保留真实当时路径；离线复算时按此表定位数据，不能把机器绝对路径当网页GPT可直接访问路径。
- 唯一ZIP的`manifest.json`含每payload成员bytes/SHA256/CRC32。manifest本身不自哈希；外部verification记录其hash和整体ZIP SHA，避免循环自哈希。`tools/final-operations/package_delivery.py verify <ZIP>`可复验路径/重复/所有成员SHA/CRC/敏感模式。整包字节通过与科研通过分别验收。

GPT_UPLOAD_REQUIRED: YES。只上传这一份最终ZIP；大型原件保留服务器。Toys producer不唯一、A2方法复现/新scratch训练、fixed32和描述统计的结论边界均详见RESULTS.md；没有声称网页GPT已完成二次审核。
'''
    target = STAGE/'README.md'; assert not target.exists(); target.write_text(readme)
    report = dict(status='ALL_COMPLETED_SCIENTIFIC_INPUTS_APPENDED_NOT_YET_FINAL_ZIP',
                  assembled_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  source_copies=records, copied_members=len(records), copied_bytes=sum(r['bytes'] for r in records),
                  reviewer_capture_inputs_mapped=len(aliases),
                  generated_members=['assembly/reviewer-input-path-map.json', 'README.md'],
                  publication_and_final_stage_review_still_required=True, final_ZIP_built=False)
    (STAGE/'assembly/final-inputs.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    files = [p for p in STAGE.rglob('*') if p.is_file()]
    print(json.dumps(dict(status=report['status'], staging=str(STAGE), total_payload_members=len(files),
                         total_payload_bytes=sum(p.stat().st_size for p in files),
                         reviewer_capture_inputs_mapped=len(aliases)), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
