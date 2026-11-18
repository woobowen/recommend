import hashlib,json,subprocess,urllib.parse,urllib.request
commit='84f200258ab66b33bcf1fe4104fa44ad43737e4a'
paths=['README.md','GRID baseline/experiments/beauty_rkmeans_qwen_20261010_012253/semantic_ids.pt','GRID baseline/experiments/beauty_rkmeans_qwen_20261010_012253/sid_generation/pickle/merged_predictions_tensor.pt','GRID baseline/_运行/run_pipeline.py','GRID baseline/src/utils/tensor_utils.py']
repo='/home/addaswsw/recommend-worktrees/sid-extensions-p0'; result=[]
for p in paths:
 url='https://raw.githubusercontent.com/woobowen/recommend/'+commit+'/'+urllib.parse.quote(p,safe='/')
 req=urllib.request.Request(url,headers={'Accept':'application/vnd.github.raw+json','User-Agent':'sid-audit'})
 with urllib.request.urlopen(req,timeout=30) as r: remote=r.read()
 local=subprocess.check_output(['git','show',commit+':'+p],cwd=repo)
 result.append({'path':p,'github_sha256':hashlib.sha256(remote).hexdigest(),'bytes':len(remote),'matches_git_blob_bytes':remote==local})
assert all(x['matches_git_blob_bytes'] for x in result)
print(json.dumps({'fixed_source_commit':commit,'files':result,'note':'For LFS files this verifies pointers; separately fetched true LFS payloads are checked by fetch_lfs.py.'},indent=2))
