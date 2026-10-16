"""Build a new final evidence ZIP from fixed Git content and existing raw assets.

No model calls, new statistics, or regeneration of images. The prior reviewed
snapshot remains intact and its original matrix members are reused byte-for-byte.
"""
import argparse,datetime as dt,gzip,hashlib,io,json,re,stat,subprocess,zipfile
from pathlib import Path,PurePosixPath
R=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
S=Path('/home/addaswsw/recommend-support/grid-formal-20261015')
D=S/'deliveries'
BASE=D/'grid-formal-20261016-gpt-audit-snapshot.zip'
BASE_SHA='b3627f2acaf70406da399f531fefb10471eb91b06c017049e0e81e26dd274f58'
PREFIX='manifests/grid-formal-20261015/'
def sha(b):return hashlib.sha256(b).hexdigest()
def js(x):return (json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def git(*args):return subprocess.check_output(['git','--no-optional-locks',*args],cwd=R)
def safe(n):
 p=PurePosixPath(n);assert not p.is_absolute() and '..' not in p.parts and '\\' not in n and not n.endswith('/')
 assert not any(x in ('.git','.ssh','__pycache__') for x in p.parts)
def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['content','delivery']);a=p.parse_args()
 publication=S/('publication-'+a.phase+'-20261017.json');readback=S/('publication-readback-'+a.phase+'-20261017.json')
 pub=json.loads(publication.read_text());rb=json.loads(readback.read_text());commit=pub['commit']
 assert rb['status']=='FIXED_GITHUB_FULL_BODY_READBACK_PASS' and rb['commit']==commit
 assert git('rev-parse','HEAD').decode().strip()==commit and git('ls-remote','origin','refs/heads/wbw').decode().split()[0]==commit
 assert sha(BASE.read_bytes())==BASE_SHA
 members={};sources={}
 def add(n,b,source):
  safe(n);assert n not in members,n;assert len(b)<16*1024**2,n
  members[n]=b;sources[n]=source
 def file(n,path):
  assert path.is_file() and not path.is_symlink();s=path.stat();b=path.read_bytes();assert path.stat().st_mtime_ns==s.st_mtime_ns
  add(n,b,{'kind':'existing_local_evidence','path':str(path),'sha256':sha(b)})
 # Original raw evidence, not merely an old ZIP renamed as final.
 with zipfile.ZipFile(BASE) as z:
  old=json.loads(z.read('MANIFEST.json'));index={x['path']:x for x in old['files']}
  for n in z.namelist():
   b=z.read(n)
   if n!='MANIFEST.json':assert sha(b)==index[n]['sha256'] and len(b)==index[n]['bytes']
   add('original-evidence/'+n,b,{'kind':'previous_GPT_reviewed_snapshot_member','snapshot_sha256':BASE_SHA,'member':n,'original_member_sha256':sha(b),'historical_readme_and_status_not_current':n in ['README.md','STATUS_CHECK_20261016.md','MANIFEST.json','PACKAGE_CHECKS.json'] or n.startswith('historical-stage/')})
 # Exact current Git files preserve normal relative navigation of the report.
 current=git('ls-tree','-r','--name-only',commit,'--',PREFIX).decode().splitlines()
 old_beauty=['manifests/beauty-extensions/'+n for n in ['final-review.json','b-attention-review.json','source-review.json','release-identity.json','mlp-sid-numeric-audit.json']]
 references=[]
 for path in sorted(set(current+old_beauty)):
  b=git('show',commit+':'+path)
  add('current/'+path,b,{'kind':'fixed_GitHub_commit_file','commit':commit,'path':path,'git_blob':git('rev-parse',commit+':'+path).decode().strip()})
  references.append({'commit':commit,'path':path,'url':'https://github.com/woobowen/recommend/blob/'+commit+'/'+path,'bytes':len(b),'sha256':sha(b)})
 # All least-three PNG cases from both paths, chosen before viewing images.
 report=json.loads(members['current/'+PREFIX+'attention-statistics/attention-descriptive-statistics.json'])
 plan=json.loads((S/'attention-statistics-20261017/plan.json').read_text())
 inventory=json.loads(members['original-evidence/references/PNG_INVENTORY.json'])
 for j in report['jobs']:
  base=Path(plan['jobs'][j['job_id']]);selected={r['plot_relative'] for rows in j['least_similar_three_under_declared_mean_cosine'].values() for r in rows}
  for rel in sorted(selected):
   n='original-evidence/jobs/'+j['job_id']+'/'+rel
   if n not in members:file(n,base/rel)
 for row in inventory:
  n='original-evidence/jobs/'+row['job_id']+'/output/'+f"batch-{row['batch']:02d}"+'/figures/'+row['file']
  row['included_in_final_package']=n in members
  if n in members:assert sha(members[n])==row['sha256']
 add('references/FINAL_PNG_INVENTORY.json',js(inventory),{'kind':'existing PNG identities plus final membership; no image changes'})
 add('references/FIXED_GITHUB_REFERENCES.json',js(references),{'kind':'current fixed version navigation'})
 for path in [publication,readback,S/'publication-content-20261017.json',S/'publication-readback-content-20261017.json',S/'final-resumption-20261017.json',S/'final-integration-review-20261017.json',S/'AUDIT_SNAPSHOT_TEMP_CLEANUP_RECEIPT.md']:
  n='delivery-evidence/'+path.name
  if n not in members:file(n,path)
 for path in [S/'final-delivery-independent-review-20261017.json',D/'grid-formal-20261017-final-candidate.zip.validation.json']:
  if path.exists():file('delivery-evidence/'+path.name,path)
 for path in [S/'publish_final_stage_20261017.py',S/'readback_final_stage_20261017.py',S/'check_final_identity_20261017.py',Path(__file__)]:
  file('execution-scripts/'+path.name,path)
 required=[]
 for j in report['jobs']:
  for i in range(4):required += ['original-evidence/jobs/'+j['job_id']+'/output/'+f'batch-{i:02d}/'+n for n in ['bundle/record.json','bundle/arrays.npz','figures/plot_receipt.json','off_on_checks.json']]
  required += ['current/'+PREFIX+'attention-results-review/'+j['job_id']+'-review.json']
 for n in ['attention-results-review/attention-statistics-runtime-review.json','attention-results-review/attention-statistics-source-provenance-review-20261017.json','final-integration-review-20261017.json','FINAL_VERSION_MAP.json','FINAL_RECEIPT.md','EXECUTION_SUMMARY.json']:
  required.append('current/'+PREFIX+n)
 assert all(n in members for n in required)
 assert len(report['jobs'])==8 and sum(len(j['per_sample_query_pairs']) for j in report['jobs'])==3072
 runtime=json.loads(members['current/'+PREFIX+'attention-results-review/attention-statistics-runtime-review.json'])
 assert runtime['report']['sha256']==sha(members['current/'+PREFIX+'attention-statistics/attention-descriptive-statistics.json'])
 version=json.loads(members['current/'+PREFIX+'FINAL_VERSION_MAP.json']);assert len(version['job_map'])==18
 for j in report['jobs']:
  for item in j['input_files']:
   suffix=item['path'].split('/retrieved/',1)[1]
   assert sha(members['original-evidence/jobs/'+j['job_id']+'/'+suffix])==item['sha256']
 phase_status='FINAL_CONTENT_ACCEPTED_PENDING_READINESS_METADATA' if a.phase=='content' else 'READY_FOR_REVIEW'
 readme=f'''# GRID 最终审核包（{a.phase}）

状态：{phase_status}。固定发布提交：{commit}；最终网页GPT二次审核仍待进行。
此包由已审原件加本轮真实统计/审核/发布证据重新构建，不是旧包改名。

阅读顺序：
1. current/{PREFIX}FINAL_RECEIPT.md、EXECUTION_SUMMARY.json：当前统一结论。
2. current/{PREFIX}attention-statistics/attention-descriptive-statistics.json：8组×32人×6对×2路径=3072记录，完整逐样本/汇总/least-three。
3. current/{PREFIX}attention-results-review/attention-statistics-runtime-review.json 与来源补审：真实非作者从NPZ独立复算；final-integration-review-20261017.json为整合审核。
4. original-evidence/jobs：18个核心job的原request/status/cleanup/RESULT与既有来源；八组新增32份NPZ/record全部保留。original-evidence/reviews与current审核目录保留旧技术审核。
5. references/FINAL_PNG_INVENTORY.json：全部256 PNG哈希/位置；每组示例和两路径最不相似三个样本的原PNG已实际纳入。
6. current/{PREFIX}FINAL_VERSION_MAP.json、ATTENTION_ARTIFACT_INDEX.json及delivery-evidence：不同真实运行commit/snapshot/资产/原结果→旧审核→本次统计→固定GitHub回读。

original-evidence/README、STATUS_CHECK及historical-stage是已审旧阶段状态，含历史PARTIAL和未完成统计，不作为当前状态。原始Beauty MLP fixed32继承，旧PASS/COMPLETE协议不同，本轮未改其RESULT或纳入第九组统计；其原结果、小样本和旧审核都保留。
新增GPU训练/测试/注意力预算和调用为0；没有改变公式、checkpoint、样本、PNG或评价器。固定32样本和原随机化条件的描述性证据不能证明完全忽略SID结构或无损并行生成。高cosine与非零TV、Sports/A低相似案例均保留。
大型模型、数据及全量预测不复制；原路径/大小/hash见原RESULT、checkpoint实体回执、collection/export manifest。必要八组矩阵没有省略。
包内MANIFEST列除自身外全部成员；ZIP整体SHA与验证/非作者验包回执置包外，避免自引用。网页外部最终审核未预签。
'''
 add('README.md',readme.encode(),{'kind':'final package navigation'})
 # Scan every textual member and decompressed JSONL archives; no script execution.
 patterns={'private_key':rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----','github_secret':rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b','openai_secret':rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}\b','aws_access_key':rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b','HF_secret':rb'\bhf_[A-Za-z0-9]{24,}\b','URL_credentials':rb'https?://[^\s/"<>:]+:[^\s/"<>@]+@'}
 findings=[]
 for n,b in members.items():
  body=gzip.decompress(b) if n.endswith('.gz') else b
  for rule,pattern in patterns.items():
   if re.search(pattern,body):findings.append((n,rule))
 assert not findings,findings
 manifest={'kind':'GRID final evidence package','phase':a.phase,'commit':commit,'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'previous_snapshot_sha256':BASE_SHA,'core_jobs':18,'main_statistics_conditions':8,'main_pair_path_records':3072,'raw_main_NPZ_count':32,'manifest_self_excluded':True,'member_count_including_manifest':len(members)+1,'files':[{'path':n,'bytes':len(b),'sha256':sha(b),'source':sources[n]} for n,b in sorted(members.items())]}
 add('MANIFEST.json',js(manifest),{'kind':'package member manifest'})
 out=D/('grid-formal-20261017-final-candidate.zip' if a.phase=='content' else 'grid-formal-20261017-final-gpt-audit.zip')
 assert not out.exists()
 with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for n,b in sorted(members.items()):
   info=zipfile.ZipInfo(n,dt.datetime.now().timetuple()[:6]);info.create_system=3;info.external_attr=(stat.S_IFREG|0o644)<<16;info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
 with zipfile.ZipFile(out) as z:
  assert len(z.namelist())==len(set(z.namelist()))==len(members) and z.testzip() is None
  stored=json.loads(z.read('MANIFEST.json'));assert set(z.namelist())=={x['path'] for x in stored['files']}|{'MANIFEST.json'}
  for row in stored['files']:
   n=row['path'];safe(n);info=z.getinfo(n);assert stat.S_IFMT(info.external_attr>>16)==stat.S_IFREG
   b=z.read(n);assert len(b)==row['bytes'] and sha(b)==row['sha256']
   if n.endswith('.npz'):
    with zipfile.ZipFile(io.BytesIO(b)) as inner:assert inner.testzip() is None
  assert stored['member_count_including_manifest']==len(z.namelist())
 checksum=sha(out.read_bytes())
 with out.with_suffix('.zip.sha256').open('x') as f:f.write(checksum+'  '+out.name+'\n')
 receipt={'status':'FINAL_PACKAGE_INTEGRITY_PASS_PENDING_NONAUTHOR_PACKAGE_REVIEW','phase':a.phase,'commit':commit,'zip_path':str(out),'windows_path':subprocess.check_output(['wslpath','-w',str(out)],text=True).strip(),'bytes':out.stat().st_size,'members':len(members),'sha256':checksum,'CRC_all_members_read':True,'manifest_size_SHA_paths_exact':True,'unsafe_duplicate_symlink_paths':False,'sensitive_pattern_findings':findings,'core_jobs':18,'main_NPZ':32,'pair_path_records':3072,'actual_statistics_SHA256':runtime['report']['sha256'],'all_256_PNG_inventory_retained':True,'included_new_attention_PNGs':sum(x['included_in_final_package'] for x in inventory),'large_assets_referenced_not_missing':True,'previous_stage_zip_preserved':sha(BASE.read_bytes())==BASE_SHA,'completed_at_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
 with out.with_suffix('.zip.validation.json').open('x') as f:f.write(js(receipt).decode())
 print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
