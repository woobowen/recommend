"""Serial normal wbw publication; remote content readback recorded separately."""
import argparse,datetime as dt,hashlib,json,subprocess
from pathlib import Path
R=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
S=Path('/home/addaswsw/recommend-support/grid-formal-20261015')
PREFIX='manifests/grid-formal-20261015/'
def git(*a):return subprocess.check_output(['git','--no-optional-locks',*a],cwd=R,text=True).strip()
def run(*a):
 p=subprocess.run(['git',*a],cwd=R,text=True,capture_output=True,timeout=120)
 return {'argv':['git',*a],'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['content','delivery']);a=parser.parse_args()
 out=S/('publication-'+a.phase+'-20261017.json');assert not out.exists()
 assert git('branch','--show-current')=='wbw'
 before=git('rev-parse','HEAD');assert git('ls-remote','origin','refs/heads/wbw').split()[0]==before
 assert not git('diff','--cached','--name-only')
 tracked=git('diff','--name-only').splitlines();new=git('ls-files','--others','--exclude-standard','--',PREFIX).splitlines()
 assert all(n.startswith(PREFIX) for n in tracked)
 paths=sorted(set(tracked+new));assert paths
 assert all((R/n).is_file() and not (R/n).is_symlink() for n in paths)
 log=[run('add','--',*paths)];assert log[-1]['returncode']==0
 staged=git('diff','--cached','--name-only').splitlines();assert set(staged)==set(paths)
 message='Complete formal GRID attention statistics and accepted result integration' if a.phase=='content' else 'Close verified GRID formal delivery and readiness metadata'
 log.append(run('commit','-m',message));assert log[-1]['returncode']==0,log[-1]
 commit=git('rev-parse','HEAD')
 log.append(run('push','origin','HEAD:refs/heads/wbw'))
 remote=git('ls-remote','origin','refs/heads/wbw').split()[0]
 receipt={'phase':a.phase,'observed_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'branch':'wbw','parent_commit':before,'commit':commit,'remote_ref':remote,'remote_ref_verified':remote==commit,'paths':paths,'commands':log,'changed_file_identities':[{'path':n,'bytes':(R/n).stat().st_size,'sha256':hashlib.sha256((R/n).read_bytes()).hexdigest(),'git_blob':git('hash-object',str(R/n))} for n in paths],'key_file_readback':'PENDING actual GitHub connector readback; push success alone is not readback','server_mapping':'final-resumption-20261017.json records actual six immutable deployments; no server code mutation'}
 with out.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
 assert log[-1]['returncode']==0 and receipt['remote_ref_verified']
 print(json.dumps({'phase':a.phase,'commit':commit,'files':len(paths),'remote_ref_verified':True,'receipt':str(out)}))
if __name__=='__main__':main()
