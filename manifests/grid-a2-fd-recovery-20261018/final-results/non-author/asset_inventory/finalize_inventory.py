#!/usr/bin/env python3
"""Validate the aggregate against the bounded read-only probe, then seal a deliverable."""
from pathlib import Path
import collections,datetime,hashlib,json,re

OUT=Path(__file__).resolve().parent
SUPPORT=OUT.parents[1]

def receipt(path):
    p=Path(path);b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

m=json.loads((OUT/'asset-manifest.pending-stat.json').read_text())
probe=json.loads((OUT/'targeted-remote-stat.json').read_text())
stat_by_path={p['path']:(i,p) for i,p in enumerate(probe['paths'])}
assert len(stat_by_path)==len(probe['paths'])==1325
assert all(a['exists'] for a in probe['paths'])
assert not probe['missing_version_metadata']
assert probe['remote_file_writes']==probe['GPU_or_model_imports']==probe['recursive_scans']==0
errors=[]
for a in m['assets']+m['dataset_members']:
    i,s=stat_by_path[a['path']]
    if not s['regular'] or s['is_symlink'] or s['bytes']!=a['bytes']:
        errors.append({'asset':a['path'],'expected_bytes':a['bytes'],'observed':s})
    a['current_stat']={'exists':s['exists'],'regular':s['regular'],'is_symlink':s['is_symlink'],'bytes':s['bytes'],'matches_recorded_size':s['bytes']==a['bytes'],'mode':s['mode'],'captured_utc':probe['captured_utc'],'json_pointer':f'/paths/{i}','receipt':'targeted_remote_stat'}
for d in m['directories']:
    i,s=stat_by_path[d['path']]
    assert s['directory'],d['path']
    d['current_stat']={'exists':True,'directory':True,'is_symlink':s['is_symlink'],'mode':s['mode'],'json_pointer':f'/paths/{i}','receipt':'targeted_remote_stat'}
    d['directory_bytes_not_measured']=True
assert not errors,errors

m['source_receipts']['targeted_remote_stat']=receipt(OUT/'targeted-remote-stat.json')
m['source_receipts']['inventory_probe_request']=receipt(OUT/'read-only-probe-request.json')
m['source_receipts']['environment_manifest']=receipt(OUT/'environment-manifest.json')
for a in probe['small_file_hashes']:
    assert a['content_hashed']
    i,s=stat_by_path[a['path']]
    m['assets'].append({'asset_id':'environment_python_executable' if a['path'].endswith('/python3.10') else 'historical_installed_requirements_text','path':a['path'],'kind':'regular_file','bytes':a['bytes'],'sha256':a['sha256'],'roles':['environment/executable' if a['path'].endswith('/python3.10') else 'environment/legacy_requirements_receipt'],'hash_basis':'Fresh bounded read-only content hash in this inventory; explicitly small executable/metadata exception.','payload_rehashed_by_this_inventory':True,'source_receipts':[{'source':'targeted_remote_stat','json_pointer':f'/small_file_hashes/{probe["small_file_hashes"].index(a)}'}],'current_stat':{'exists':True,'regular':s['regular'],'bytes':s['bytes'],'matches_recorded_size':s['bytes']==a['bytes'],'json_pointer':f'/paths/{i}','receipt':'targeted_remote_stat'},'read_only_recovery':{'host':'new-server','action':'Reuse exact path read-only. This executable/requirements file is not a complete environment backup.'}})

configs=[
 ('Toys-A-5300','toys',SUPPORT/'reviews/toys/raw/ga-20261017-toys5300-eval-0-v1/output/used_config.json'),
 ('Toys-A-5000','toys',SUPPORT/'reviews/toys/raw/ga-20261017-toys5000-eval-0-v1/output/used_config.json'),
 ('Sports-A1','sports',SUPPORT/'reviews/sports_a1/raw/ga-20261017-sportsa1-train-v1/output/used_config.json'),
 ('Sports-A2','sports',SUPPORT/'reviews/a2_training/terminal-originals/output/used_config.json'),
]
directory_paths={d['path'] for d in m['directories']}
for route,domain,p in configs:
    x=json.loads(p.read_text());key=route+'_actual_used_config'
    m['source_receipts'][key]=receipt(p)
    bindings=[]
    for loader,split in [('train_dataloader_config','training'),('val_dataloader_config','evaluation'),('test_dataloader_config','testing')]:
        folder=x['datamodule'][loader]['data_folder']
        assert folder in directory_paths and folder.endswith('/'+domain+'/'+split)
        bindings.append({'loader':loader,'split':split,'path':folder,'receipt':key,'json_pointer':'/datamodule/'+loader+'/data_folder','scope':'Actual effective loader folder identity only. Toys training loader in evaluation configuration does not establish upstream historical Toys checkpoint production.'})
    m['routes'][route]['actual_effective_loader_directories']=bindings

env=json.loads((OUT/'environment-manifest.json').read_text())
env['post_recovery_current_critical_versions']=probe['current_versions']
env['post_recovery_current_version_metadata_site']=probe['version_metadata_site']
env['post_recovery_probe_receipt']=receipt(OUT/'targeted-remote-stat.json')
env['critical_versions_equal_pre_recovery_probe']=all(v['version']==env['current_pre_recovery_probe']['versions'][k] for k,v in probe['current_versions'].items())
assert env['critical_versions_equal_pre_recovery_probe']
env['requirements_file_is_not_complete_environment_image']=True
(OUT/'environment-manifest.json').write_text(json.dumps(env,ensure_ascii=False,indent=2)+'\n')
m['source_receipts']['environment_manifest']=receipt(OUT/'environment-manifest.json')
m['source_receipts']['A1_independent_result_audit']=receipt(SUPPORT/'reviews/sports_a1/audit_result.json')
m['source_receipts']['Toys_independent_result_audit']=receipt(SUPPORT/'reviews/toys/toys-independent-review.json')
m['source_receipts']['A2_final_training_review']=receipt(SUPPORT/'reviews/a2_training/final_training_review.json')
m['status']='ASSET_DIRECTORY_MAPPING_PASS_WITH_EXPLICIT_ENVIRONMENT_IMAGE_LIMIT'
m['sealed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
m['temporal_identity_statement']='This is a post-execution delivery inventory, assembled from saved actual source receipts and a bounded fresh stat. Its own SHA and current receipt hashes are not claimed to have been the manifest used during past execution. Execution source85b77f5e2a5b497fdc1f620649ba488fc783a245/snapshot22286966cd544773 is distinct from the later publication commit.'
m['execution_source_mapping']={'original_scientific_source':'b44399926c98369191c2ff5c2425377c8c07d3c8','new_A2_execution_source_commit':'85b77f5e2a5b497fdc1f620649ba488fc783a245','new_A2_execution_snapshot':'/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/snapshots/22286966cd544773','publication_commit':'Bound separately by root final delivery; not inferred from source-receipt worktree HEAD.'}
m['hash_observation_policy']['fresh_bounded_small_metadata_or_executable_hashes']=probe['small_file_hashes']
m['summary']={'model_SID_embedding_or_quantizer_files':21,'small_environment_files':2,'dataset_members':1293,'dataset_member_logical_bytes':sum(a['bytes'] for a in m['dataset_members']),'all_listed_asset_and_dataset_regular_file_sizes_match':True,'current_exact_paths_statted':1325,'current_missing_paths':0,'directory_payload_hashes_invented':0,'fresh_big_asset_payload_hashes':0,'downloaded_big_assets':0,'current_critical_package_versions_checked':9}

ids=[a['asset_id'] for a in m['assets']]
assert len(ids)==len(set(ids))==23
allfiles=m['assets']+m['dataset_members']
assert len({a['path'] for a in allfiles})==len(allfiles)
assert all(Path(a['path']).is_absolute() and '..' not in Path(a['path']).parts for a in allfiles)
assert all(isinstance(a['bytes'],int) and a['bytes']>0 and re.fullmatch('[0-9a-f]{64}',a['sha256']) for a in allfiles)
assert all(d['sha256'] is None for d in m['directories'])
assert sum(a['payload_rehashed_by_this_inventory'] for a in allfiles)==2
for route,v in m['routes'].items():
    assert v['checkpoint_asset_id'] in ids and v['sid_asset_id'] in ids

(OUT/'assets-not-copied.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
seal=receipt(OUT/'assets-not-copied.json')
validation={'status':'ASSET_INVENTORY_VALIDATION_PASS','validated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest':seal,'summary':m['summary'],'fresh_stat_matched_recorded_sizes':1314,'critical_versions_equal_pre_recovery':env['critical_versions_equal_pre_recovery_probe'],'claim_limits':m['unverified_or_limited'],'fresh_payload_hashes':'Only the 2290B requirements receipt and 17506296B Python executable were freshly hashed. All model/SID/T5/TFRecord SHA values are inherited from explicit real receipts; current stat does not replace content validation.','no_new_packages_or_global_configuration_changes':True}
(OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
rows=[]
for d in m['directories']:
    if 'dataset' in d:rows.append(f"| {d['dataset']}/{d['split']} | {d['member_count']} | {d['logical_member_bytes']} |")
report=f'''# 大资产目录映射核验

`{m['status']}`。这是事后交付清单；不把本文件 SHA 冒充过去执行时使用的 manifest。科研实现和结果验收由独立专项报告承担。

本轮聚合四路线的实际 checkpoint/SID、新 A2 两份 quantizer checkpoint、3 层原始量化码、复用 T5 编码输出与六个原 T5-small 成员，共 21 个文件。另收录 1293 个 Sports/Toys TFRecord 成员及各自路径、字节数、SHA、原 archive/member/manifest 行号恢复入口。Toys 两候选共享同一个 SID，均保留，不按指标择优。

| 数据 split | 成员数 | 逻辑字节数 |
|---|---:|---:|
{chr(10).join(rows)}

只读探针对清单给出的 1325 个明确路径调用 stat，没有递归扫描。1314 个模型/SID/T5/TFRecord 文件均存在、为独立 regular file，大小与原记录相同。模型、T5 权重和数据内容没有重新散列或下载。它们的 SHA 明确继承真实旧加载/恢复回执或本轮 runner 终态 manifest；本轮先前独立 strict-load、SID 和量化审核另有原回执，不由这次 stat 代替。

8 个目录均不声明单一目录 SHA。6 个数据目录按逐成员清单恢复；T5 目录按六个已保存成员恢复。环境目录不递归测量容量，不宣称有完整逐文件镜像。环境版本源包含 256 条历史 distribution 元数据，9 个关键库通过本轮只读 METADATA 核对与修复前记录一致；历史记录中的 python3.1 路径原样保留，不当作已验证路径。

仅对两个明确的小文件做了本轮有界 hash：2290B 的 legacy installed-requirements.txt 与 17506296B 的 Python executable。它们不构成完整可重建环境；环境目录 SHA 和完整字节大小仍为 null。没有安装包、全局配置修改、GPU/model import、重训练、恢复写入或实验提交。

限制继续保留：Toys 上游 producer 不唯一；A2 是方法复现，不是找回历史 Sports 权重；原历史 hash 的继承与当前 payload 重新测量必须区分。各路线有效配置的三个 loader 目录均已直接对应本清单，但 Toys 评价配置中的 train loader 不证明其历史 checkpoint 的上游生产过程。

交付入口：`assets-not-copied.json`，{seal['bytes']}B，SHA-256 `{seal['sha256']}`。逐路径 stat 原件为 `targeted-remote-stat.json`，机器校验为 `validation.json`；环境版本与限制为 `environment-manifest.json`。可放入最终轻量 ZIP 的根目录；大文件留在 new-server 原路径。GPT_UPLOAD_REQUIRED: YES（由主控聚合一次最终 ZIP）。
'''
(OUT/'REPORT.md').write_text(report)
print(json.dumps(validation,ensure_ascii=False))
