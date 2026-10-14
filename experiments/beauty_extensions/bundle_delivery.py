"""Build and verify the lightweight review ZIP after fixed GitHub publication."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--commit',required=True)
    p.add_argument('--base',default='5ed4e701c41d977e24c4c89db0ae1025f9faf701')
    args=p.parse_args();assert re.fullmatch('[0-9a-f]{40}',args.commit)
    repo=Path(__file__).resolve().parents[2]
    support=Path('/home/addaswsw/recommend-support/sid-extensions-p0/beauty-extensions')
    collection=Path(json.loads((support/'collection-receipt.json').read_text())['directory'])
    publication=json.loads((support/'publication.json').read_text());assert publication['commit']==args.commit and publication['remote_ref_verified']
    names=subprocess.check_output(['git','diff','--name-only',args.base,args.commit],cwd=repo,text=True).splitlines()
    summary=json.loads(subprocess.check_output(['git','show',args.commit+':manifests/beauty-extensions/EXECUTION_SUMMARY.json'],cwd=repo))
    members={}
    for name in names:
        members['repository/'+name]=subprocess.check_output(['git','show',args.commit+':'+name],cwd=repo)
    for f in collection.rglob('*'):
        if f.is_file():
            assert not f.is_symlink()
            members['evidence/'+str(f.relative_to(collection))]=f.read_bytes()
    for name in ['AGENTS.md','PROJECT_RULES.md','REMOTE_EXECUTION.md']:
        members['applied-rules/'+name]=(Path('/home/addaswsw/recommend')/name).read_bytes()
    for name in ['publication.json','terminal-states.json','collection-receipt.json']:
        members['external-receipts/'+name]=(support/name).read_bytes()
    for f in (support/'observations').rglob('*.json'):
        assert f.stat().st_size<256*1024 and not f.is_symlink()
        members['external-receipts/observations/'+str(f.relative_to(support/'observations'))]=f.read_bytes()
    provenance=Path('/home/addaswsw/recommend-support/wmh-a800-audit-20261010')
    original=provenance/'preservation/A800-grid-20261010-v1/restored/A800_GRID'
    extra=[original/'logs/inference/runs/2025-09-15/20-54-16/.hydra/config.yaml',
           original/'logs/inference/runs/2025-09-21/16-36-33/.hydra/config.yaml',
           original/'logs/inference/runs/2025-09-15/19-47-36/.hydra/config.yaml',
           provenance/'analysis/a800-baseline/artifact-inspection-b05-v1/metadata/21b174d051548d28eadc995c365c6c7beaf7410535fe726b8f720e9555a04e00.json',
           provenance/'analysis/a800-baseline/artifact-inspection-b03-b04-v1/metadata/095f97fb5c6add6c023dc0c4b746def9725757027dec95397d0d4de44135ec9a.json']
    source_review=json.loads((repo/'manifests/beauty-extensions/source-review.json').read_text())
    inherited=source_review['mlp_budget_and_approval_supplement']['exact_packaging_inputs']
    for key in ('metrics_csv','train_log','effective_config','restart_metadata'):
        identity=inherited[key];f=Path(identity['path'])
        assert f.is_file()
        if identity.get('sha256'):
            assert hashlib.sha256(f.read_bytes()).hexdigest()==identity['sha256']
        extra.append(f)
    for f in extra:
        if f.is_file():
            assert f.stat().st_size<2*1024**2
            members['inherited-source-evidence/'+str(f.relative_to(provenance))]=f.read_bytes()
    refs=['references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py',
          'references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py',
          'references/wmh-original/GRID/visualize_complete_decoder_attention.py',
          'GRID baseline/src/models/modules/semantic_id/tiger_generation_model.py',
          'GRID baseline/src/data/loading/components/collate_functions.py',
          'GRID baseline/src/data/loading/components/label_function.py',
          'GRID baseline/src/components/eval_metrics.py']
    for name in refs:
        members['fixed-originals/'+name]=subprocess.check_output(['git','show',args.commit+':'+name],cwd=repo)
    readme=f'''# Beauty 一次性工作审核包 — PARTIAL_BLOCKED

固定GitHub提交：{args.commit}
入口：https://github.com/woobowen/recommend/blob/{args.commit}/docs/sid-extensions/experiments/shuffle.md

先读 repository/manifests/beauty-extensions/EXECUTION_SUMMARY.json 和 final-review.json、b-attention-review.json。
三份原实验说明包含本轮采用值、批准状态、工程结果及精确未完成项。本轮正式训练结果见EXECUTION_SUMMARY.formal_results；六组A/B正式评测仍NOT_RUN，不能当Beauty全部实验完成证明。

evidence/ 含{len(summary['jobs'])}个job原request/status/log/必要输出（包括失败），没有checkpoint或原TFRecord载荷。
beauty-b-smoke-r3/output/trainer-test-0、3、4 是各19用户的工程预测，后续尾部device mismatch使整个job FAILED。
beauty-preflight-metrics-verification-v1/output/ 是CPU原evaluator复算；19人四项均0不作研究结论。
beauty-a-metadata-r3、beauty-b-smoke-r3、beauty-mlp-smoke-r3 的 fixed_attention_samples.json 和 test_target_index.jsonl 保存固定32及22,363目标身份。
beauty-a-attention-r3、beauty-b-attention-r2、beauty-mlp-attention-engineering-r3 的 output/small-attention/bundle/record.json + arrays.npz 与 figures/ 是仅Normal的3用户工程采集；MLP只有3次optimizer更新。
beauty-mlp-evaluation-index-v1 与 beauty-mlp-accum64-fullval-v1 是用户批准后真实acc64/完整validation资源检查；beauty-mlp-fixed8-attention-v1 仍用3-update工程模型。
如有 beauty-mlp-formal-v1，output/training_result.json区分正式完成/中断、最佳checkpoint、训练终点及完整test/复算；如有 beauty-mlp-formal-attention32-v1，output/RESULT.json绑定完成的新MLP最佳权重和事前固定的32用户。相应CPU曲线job不运行模型。

repository/ 的源码来自实际提交字节；external-receipts/publication.json 证明GitHub回读及服务器源码映射。
inherited-source-evidence/ 是必要旧配置/惰性checkpoint元数据，不能替代本轮实测；fixed-originals/ 保留所引用原正文。
applied-rules/ 是显式读取的本地v1.1规范，未改全局Codex配置。

MANIFEST.json列出每个文件大小和SHA256；ZIP整体SHA在包外。全部矩阵为numeric NPZ，读取须allow_pickle=False。
大型权重保留new-server，恢复入口为各request及EXECUTION_SUMMARY中的路径；不重新生成SID/embedding、不开Parallel/跨域。
'''
    members['README.md']=readme.encode()
    secret=re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{30,}')
    records=[]
    for name,data in sorted(members.items()):
        path=Path(name);assert not path.is_absolute() and '..' not in path.parts
        assert path.suffix not in ('.ckpt','.pt','.pyc','.pkl','.tfrecord')
        if path.suffix in ('.py','.json','.jsonl','.yaml','.md','.csv','.log'):
            assert not secret.search(data),name
        records.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    members['MANIFEST.json']=(json.dumps({'commit':args.commit,'base_commit':args.base,'status':summary['status'],'files':records,
        'zip_hash_stored_outside':True,'secret_pattern_scan':'PASS'},ensure_ascii=False,indent=2)+'\n').encode()
    dest=support/'deliveries';dest.mkdir(exist_ok=True)
    archive=dest/'beauty-extensions-partial-20261014.zip';assert not archive.exists()
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(members.items()):z.writestr(name,data)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(members)
        manifest=json.loads(z.read('MANIFEST.json'))
        assert set(z.namelist())=={r['path'] for r in manifest['files']}|{'MANIFEST.json'}
        for r in manifest['files']:
            data=z.read(r['path']);assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    (archive.with_suffix('.zip.sha256')).write_text(digest+'  '+archive.name+'\n')
    unc=subprocess.check_output(['wslpath','-w',str(archive)],text=True).strip()
    receipt={'path':str(archive),'windows_unc':unc,'bytes':archive.stat().st_size,'sha256':digest,
        'members':len(members),'crc_and_member_sha256_verified':True,'secret_pattern_scan':'PASS','commit':args.commit}
    (dest/'zip-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
