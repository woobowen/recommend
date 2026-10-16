"""Close current readiness metadata only after real content publication and package review."""
from pathlib import Path
import json,hashlib,datetime as dt
R=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
S=Path('/home/addaswsw/recommend-support/grid-formal-20261015')
F=R/'manifests/grid-formal-20261015'
def identity(p):
 b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def copy(src,dst):
 assert not dst.exists();dst.write_bytes(src.read_bytes())
publication=json.loads((S/'publication-content-20261017.json').read_text())
readback=json.loads((S/'publication-readback-content-20261017.json').read_text())
review_path=S/'final-delivery-independent-review-20261017.json'
review=json.loads(review_path.read_text())
candidate=json.loads((S/'deliveries/grid-formal-20261017-final-candidate.zip.validation.json').read_text())
assert 'PASS' in review['status'] and candidate['sha256'] in json.dumps(review)
assert publication['commit']==readback['commit']==candidate['commit']
assert readback['status']=='FIXED_GITHUB_FULL_BODY_READBACK_PASS'
for src,name in [(S/'publication-content-20261017.json','publication-content-20261017.json'),(S/'publication-readback-content-20261017.json','publication-readback-content-20261017.json'),(review_path,'final-delivery-independent-review-20261017.json'),(S/'deliveries/grid-formal-20261017-final-candidate.zip.validation.json','final-candidate-package-validation-20261017.json')]:
 copy(src,F/name)
summary=json.loads((F/'EXECUTION_SUMMARY.json').read_text())
assert summary['status']=='COMPLETED_PENDING_DELIVERY_REVIEW'
summary['status']='READY_FOR_REVIEW';summary['remaining_work']=[]
summary['historical_stage_jobs_snapshot']=summary['jobs']
fresh=json.loads((S/'final-resumption-20261017.json').read_text())['result']['jobs']
assert len(fresh)==18 and len({x['job_id'] for x in fresh})==18 and all(x['status']=='SUCCEEDED' and x['exit_code']==0 for x in fresh)
summary['jobs']=fresh
summary['independent_review_scope']['final_integration']='final-integration-review-20261017.json'
summary['independent_review_scope']['final_package']='final-delivery-independent-review-20261017.json: actual C1 candidate package; final metadata delta package final readback/review receipt is outside ZIP to avoid self-reference'
summary['finalization_20261017']['closure']={'status':'READY_FOR_REVIEW','content_commit':publication['commit'],'content_GitHub_full_body_readback':identity(S/'publication-readback-content-20261017.json'),'content_integration_review':identity(S/'final-integration-review-20261017.json'),'candidate_package_validation':identity(S/'deliveries/grid-formal-20261017-final-candidate.zip.validation.json'),'candidate_independent_delivery_review':identity(review_path),'metadata_transition_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'final_delivery_commit_and_ZIP_identity':'Recorded in final external publication/ZIP/independent validation receipts; not self-referenced in this commit','final_external_GPT_review':'PENDING; prior stage review does not pre-approve new statistics/final delivery'}
save(F/'EXECUTION_SUMMARY.json',summary)
p=F/'FINAL_RECEIPT.md';body=p.read_text()
old='**COMPLETED_PENDING_DELIVERY_REVIEW**：六组Beauty正式全量评测、三域MLP正式成果、八组新增fixed32注意力以及本轮3072条CPU描述统计和非作者实际复算均完成。GitHub发布/回读、最终ZIP及其独立验包尚待本轮后续步骤；此处不预签这些检查。'
new='**READY_FOR_REVIEW**：六组Beauty正式全量评测、三域MLP正式成果、八组新增fixed32注意力及3072条CPU统计、非作者统计与整合审核、内容提交实际GitHub回读和交付候选包独立验包均已完成。最终交付元数据提交和最终封装的真实SHA/回读/验包记录在包外回执与最终回复中绑定，不在本文自引用；最终网页GPT二次审核待进行。'
assert old in body;body=body.replace(old,new)
body=body.replace('最终整合报告审核与包审核将另存真实回执，不预签。','最终整合非作者审核已通过，见 [整合审核](final-integration-review-20261017.json)；已发布内容候选包也通过真实非作者验包，见 [交付候选审核](final-delivery-independent-review-20261017.json)。最终元数据封装的包外独立回执随最终交付提供；不把旧审核冒充新执行。')
body=body.replace('当前发布与最终验包仍待执行，不将未执行检查写成PASS。','已完成内容提交 `'+publication['commit']+'` 的正常发布与六个关键文件完整正文回读，见 [实际回读](publication-readback-content-20261017.json)。本次最终元数据HEAD和最终封装的实测身份在包外回执给出；不改变已审研究数字/源码、统计输出或原件。')
body=body.replace('正式研究结果与CPU统计已验收；本轮尚须完成最终整合非作者审核、正常发布/实际回读和最终ZIP独立校验。完成后再将当前回执更新为READY_FOR_REVIEW，网页最终二次审核仍待进行。','本次授权范围无剩余必做项。研究结论的外推限制仍保留，等待网页GPT对新统计与最终交付二次审核；不自动开展Parallel、扩域或新增训练。')
p.write_text(body)
for src in [Path(__file__),S/'publish_final_stage_20261017.py',S/'readback_final_stage_20261017.py',S/'deliveries/package_final_stage_20261017.py']:
 copy(src,F/src.name)
print(json.dumps({'status':'READY_METADATA_PREPARED_FROM_ACTUAL_REVIEWED_CONTENT','content_commit':publication['commit'],'summary':identity(F/'EXECUTION_SUMMARY.json'),'report':identity(p)}))
