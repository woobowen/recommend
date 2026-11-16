"""Synthetic single-process test of real Control.step, no Runner subprocess."""
import datetime, json, os, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import production_control as c
from formal_control import digest

class Runner:
    def __init__(self, status):self.status=status;self.submitted=[]
    def query(self, job):return self.status if job=='old' else None
    def submit(self,path):
        req=json.loads(Path(path).read_text());self.submitted.append(req)
        return {'status':'RUNNING','request_sha256':digest(req)}

class BudgetStep(unittest.TestCase):
    def test_retry_uses_same_attempt_ledger(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_CPU_') as tmp:
            root=Path(tmp)/'SYNTHETIC_CPU';root.mkdir(); plan_path=root/'plan.json'
            plan={'root':str(root),'evidence_scope':c.SYNTHETIC,'run_commit':'a'*40,
                  'runner':'/unused','runner_sha256':'a'*64,'job_root':'/unused',
                  'temporary_root':str(root),'task_id':'synthetic',
                  'budget':{'cpu':3600,'train_normal':10000,'s3':1000,'s4':1000,'total':15600,'cleanup_reserve':120},
                  'stage_seconds':{s:600 for s in c.STAGES}}
            plan_path.write_text(json.dumps(plan))
            old={'stage':'collect','attempt':0,'job_id':'old','request_sha256':'x',
                 'status':'FAILED','charged_seconds':0}
            stages={s:{'status':'COMPLETE'} for s in c.STAGES[:-1]}
            stages['collect']={'status':'SUBMITTING','output':str(root/'old'),'attempt_index':0}
            (root/'state.json').write_text(json.dumps({'attempts':[old],'stages':stages,'status':'FAILED'}))
            status={'request_sha256':'x','status':'FAILED','started_at':'2026-01-01T00:00:00+00:00',
                    'ended_at':'2026-01-01T00:03:20+00:00','worker_alive':False,
                    'workgroup_live':False,'cleanup_verified':True,'exit_code':1}
            runner=Runner(status)
            def factory(plan,stage,n,seconds,path,*args):
                return {'job_id':'retry','timeout_seconds':seconds},str(root/'new')
            original_read=Path.read_text
            def fake_meminfo(path,*args,**kwargs):
                if str(path)=='/proc/meminfo': return 'MemAvailable: 100000000 kB\n'
                return original_read(path,*args,**kwargs)
            with patch.object(c,'validate_plan',side_effect=lambda p:p), patch.object(Path,'read_text',fake_meminfo):
                ctl=c.Control(plan_path,request_factory=factory,runner=runner)
                with self.assertRaisesRegex(ValueError,'explicit retry'):ctl.step()
                state=ctl.step(retry='collect')
            self.assertEqual(state['attempts'][0]['charged_seconds'],200)
            self.assertEqual(runner.submitted[0]['timeout_seconds'],280)
            self.assertEqual(len(state['attempts']),2)

if __name__=='__main__':unittest.main(verbosity=2)
