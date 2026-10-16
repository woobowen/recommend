"""Read fixed GitHub file bodies after push and compare complete byte identities."""
import argparse,concurrent.futures,datetime as dt,hashlib,json,subprocess,urllib.request,urllib.parse
from pathlib import Path
R=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
S=Path('/home/addaswsw/recommend-support/grid-formal-20261015')
PREFIX='manifests/grid-formal-20261015/'
def git(*args):return subprocess.check_output(['git','--no-optional-locks',*args],cwd=R)
def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['content','delivery']);a=p.parse_args()
 publication=json.loads((S/('publication-'+a.phase+'-20261017.json')).read_text());commit=publication['commit']
 assert git('ls-remote','origin','refs/heads/wbw').decode().split()[0]==commit
 keys=['FINAL_RECEIPT.md','EXECUTION_SUMMARY.json','FINAL_VERSION_MAP.json','attention-results-review/attention-statistics-runtime-review.json','attention-statistics/attention-descriptive-statistics.json','final-integration-review-20261017.json']
 def check(n):
  path=PREFIX+n;expected=git('show',commit+':'+path)
  url='https://raw.githubusercontent.com/woobowen/recommend/'+commit+'/'+urllib.parse.quote(path)
  with urllib.request.urlopen(url,timeout=40) as response:
   body=response.read();status=response.status
  assert status==200 and body==expected,path
  blob=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
  assert blob==git('rev-parse',commit+':'+path).decode().strip()
  return {'path':path,'url':url,'http_status':status,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),'git_blob':blob,'full_body_exact':True}
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(check,keys))
 receipt={'status':'FIXED_GITHUB_FULL_BODY_READBACK_PASS','commit':commit,'branch':'wbw','remote_ref_actual':git('ls-remote','origin','refs/heads/wbw').decode().split()[0],'observed_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'files':rows,'note':'GitHub connector also read content-phase metadata/small files; oversized statistics body verified through fixed raw GitHub HTTPS GET, no default-branch ambiguity','server_mapping':'final-resumption-20261017.json; immutable historical jobs not overwritten'}
 out=S/('publication-readback-'+a.phase+'-20261017.json')
 with out.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({'status':receipt['status'],'commit':commit,'files':len(rows),'receipt':str(out)}))
if __name__=='__main__':main()
