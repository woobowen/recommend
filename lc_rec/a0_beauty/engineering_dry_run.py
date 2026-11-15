#!/usr/bin/env python3
"""Pure simulation of serial Beauty dependencies; cannot submit or write jobs."""
import json
STAGES=('train','normal','normal_eval','s3','s3_eval','s4','s4_eval','paired')
ACTIVE={'ACCEPTED','RUNNING','CONTROL_LOST'}
TERMINAL_BAD={'FAILED','TIMED_OUT','CLEANUP_UNCONFIRMED'}

def allowed(stage,states,authorized):
 if stage not in STAGES:raise ValueError('unknown stage')
 if not authorized:return False,'FORMAL_AUTHORIZATION_REQUIRED'
 current=states.get(stage,{})
 if current.get('state') in ACTIVE:return False,'EXISTING_OR_UNCONFIRMED_JOB'
 if current.get('state') in TERMINAL_BAD:return False,'EXPLICIT_RECOVERY_REQUIRED'
 if current.get('state')=='COMPLETE':return False,'ALREADY_COMPLETE'
 if current.get('state') not in (None,'NOT_SUBMITTED'):return False,'UNKNOWN_STATE_FAIL_CLOSED'
 index=STAGES.index(stage)
 for required in STAGES[:index]:
  prev=states.get(required,{})
  if not (prev.get('state')=='COMPLETE' and prev.get('exit_code')==0 and prev.get('cleanup_verified') is True and prev.get('artifacts_verified') is True):return False,'DEPENDENCY_NOT_VERIFIED'
 return True,'PLAN_ONLY_NO_SUBMISSION'

def test():
 ok={'state':'COMPLETE','exit_code':0,'cleanup_verified':True,'artifacts_verified':True}
 checks={}
 def check(name,expect,*args):
  got=allowed(*args);assert got==expect,(name,got);checks[name]=got[1]
 check('without_authorization',(False,'FORMAL_AUTHORIZATION_REQUIRED'),'train',{},False)
 check('plan_train',(True,'PLAN_ONLY_NO_SUBMISSION'),'train',{},True)
 for status in ACTIVE:check('existing_'+status,(False,'EXISTING_OR_UNCONFIRMED_JOB'),'train',{'train':{'state':status}},True)
 for status in TERMINAL_BAD:check('failure_'+status,(False,'EXPLICIT_RECOVERY_REQUIRED'),'train',{'train':{'state':status}},True)
 for status in ('UNCONFIRMED','UNKNOWN','MYSTERY'):check('unknown_'+status,(False,'UNKNOWN_STATE_FAIL_CLOSED'),'train',{'train':{'state':status}},True)
 check('duplicate_complete',(False,'ALREADY_COMPLETE'),'train',{'train':ok},True)
 for stage in STAGES[1:]:
  previous=STAGES[STAGES.index(stage)-1]
  completed={name:ok for name in STAGES[:STAGES.index(stage)]}
  check('missing_dep_'+stage,(False,'DEPENDENCY_NOT_VERIFIED'),stage,{},True)
  for field in ('exit_code','cleanup_verified','artifacts_verified'):
   bad=ok.copy();bad[field]=None
   check('bad_'+stage+'_'+field,(False,'DEPENDENCY_NOT_VERIFIED'),stage,{**completed,previous:bad},True)
  check('ready_'+stage,(True,'PLAN_ONLY_NO_SUBMISSION'),stage,completed,True)
 # A failed CPU stage leaves its upstream GPU stage COMPLETE; there is no transition that resubmits it.
 states={'train':ok,'normal':ok,'normal_eval':{'state':'FAILED'}}
 check('eval_failure_no_regeneration',(False,'ALREADY_COMPLETE'),'normal',states,True)
 check('eval_failure_stops_shuffle',(False,'DEPENDENCY_NOT_VERIFIED'),'s3',states,True)
 return {'status':'PASS_SIMULATION_ONLY','checks':checks,'stage_order':STAGES,'formal_submit_implemented':False,'recovery_not_implemented':True}
if __name__=='__main__':print(json.dumps(test(),indent=2))
