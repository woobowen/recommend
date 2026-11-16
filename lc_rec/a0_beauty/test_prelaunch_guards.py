"""Bounded CPU-only checks of production guards; all approvals are synthetic."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import formal_train
import production_control as control
from formal_control import digest
from production_identity import identity

class PrelaunchGuards(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='SYNTHETIC_PRELAUNCH_GUARDS_')
        self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name)
        self.registry=patch.object(control,'EXPERIMENT_REGISTRY',self.base/'registry')
        self.registry.start(); self.addCleanup(self.registry.stop)
        release=self.base/'release'; train=release/'lc_rec/a0_beauty/formal_train.py'
        train.parent.mkdir(parents=True); train.write_text('synthetic source')
        evidence=self.base/'evidence'; evidence.write_text('synthetic evidence')
        review=self.base/'review.json'; precision=self.base/'precision.json'
        self.plan={'scope':'FORMAL_SUBMIT','evidence_scope':'FORMAL','task_id':'fixture',
          'root':str(self.base/'root'),'job_root':str(self.base/'jobs'), 'run_commit':'a'*40,
          'prediction_run_commit':'a'*40,'config_sha256':'b'*64,'asset_sha256':{'fixture':'sha'},
          'model':str(self.base/'revision'),'release':str(release), 'precision_approval':str(precision),
          'prelaunch_review':str(review),'budget':{'fixture':120}, 'python':'/usr/bin/python3',
          'namespace':{'host':'fixture','user':'fixture','uid':1,'namespace':'fixture','deployment_sha256':'0'*64},
          'env':{'CUDA_VISIBLE_DEVICES':'0,1'},'inputs':[], 'runner':str(self.base/'runner'),
          'runner_sha256':'1'*64,'export_manifest':{'path':str(self.base/'manifest'),'sha256':'2'*64}}
        review.write_text(json.dumps({'status':'PASS','RUN_COMMIT':self.plan['run_commit'],
          'config_sha256':self.plan['config_sha256'],'formal_train_sha256':control.sha(train),
          'review_kind':'INDEPENDENT_NONAUTHOR','reviewer':'SYNTHETIC_FIXTURE',
          'evidence_files':[identity(evidence)]}))
        precision.write_text(json.dumps({'formal_a_only_bf16_authorized':True,'scope':'SYNTHETIC_FIXTURE'}))
        self.plan['approval_files']={'prelaunch_review':identity(review),'precision_approval':identity(precision)}
        self.plan_path=self.base/'plan.json'; self.plan_path.write_text(json.dumps(self.plan))
        root=Path(self.plan['root']);root.mkdir(); (root/'state.json').write_text('{}')

    def test_approval_exact_bytes_and_worker_read(self):
        control.approval_documents(self.plan,require_pass=True)
        for key in ('prelaunch_review','precision_approval'):
            row=self.plan['approval_files'][key]
            self.assertTrue(formal_train.approval_json(row['path'],row['sha256']))
            original=Path(row['path']).read_bytes()
            Path(row['path']).write_bytes(original+b' ')
            with self.assertRaises(ValueError):control.approval_documents(self.plan,require_pass=True)
            with self.assertRaises(ValueError):formal_train.approval_json(row['path'],row['sha256'])
            Path(row['path']).write_bytes(original)
        row=self.plan['approval_files']['prelaunch_review']; target=Path(row['path'])
        target.unlink(); target.symlink_to(self.plan['precision_approval'])
        with self.assertRaises((ValueError,OSError)):control.approval_documents(self.plan,True)
        with self.assertRaises(OSError):formal_train.approval_json(row['path'],row['sha256'])

    def test_missing_swap_and_request_inputs(self):
        request,_=control.request(self.plan,'train',0,90,self.plan_path)
        inputs={(x['path'],x['sha256']) for x in request['inputs']}
        for row in self.plan['approval_files'].values(): self.assertIn((row['path'],row['sha256']),inputs)
        swapped=copy.deepcopy(self.plan)
        swapped['approval_files']['prelaunch_review'],swapped['approval_files']['precision_approval']=(
            swapped['approval_files']['precision_approval'],swapped['approval_files']['prelaunch_review'])
        with self.assertRaises(ValueError): control.approval_documents(swapped)
        Path(self.plan['prelaunch_review']).unlink()
        with self.assertRaises((OSError,ValueError)):control.approval_documents(self.plan)
        with self.assertRaises(PermissionError):control.authorize(self.plan,None,None)

    def test_authorization_digest_plan_binding_and_expiry(self):
        receipt=self.base/'SYNTHETIC_AUTHORIZATION.json'
        doc={'scope':'FORMAL','authorization_kind':'USER_LAUNCH_AUTHORIZATION',
             'authorized_by':'user','user_message_reference':'SYNTHETIC_FIXTURE',
             'authorization_text':'SYNTHETIC_FIXTURE','action':'FORMAL_BEAUTY_TRAIN_NORMAL_S3_S4',
             'plan_sha256':digest(self.plan),'run_commit':self.plan['run_commit'],
             'config_sha256':self.plan['config_sha256'],'budget':self.plan['budget'],
             'task_id':self.plan['task_id'], 'experiment_series':control.INITIAL_SERIES, 'expires_utc':'2099-01-01T00:00:00+00:00'}
        receipt.write_text(json.dumps(doc)); expected=control.sha(receipt)
        with self.assertRaises(ValueError):control.authorize(self.plan,str(receipt),expected)
        with patch.object(control,'scope_guard'):
            self.assertTrue(control.authorize(self.plan,str(receipt),expected))
        with self.assertRaises(ValueError):control.authorize(self.plan,str(receipt),'0'*64)
        changed=copy.deepcopy(self.plan);changed['budget']={'fixture':121}
        with patch.object(control,'scope_guard'):
            with self.assertRaises(ValueError):control.authorize(changed,str(receipt),expected)
        next_plan=copy.deepcopy(self.plan);next_plan['experiment_series']='separately-authorized-series'
        doc['experiment_series']=next_plan['experiment_series'];doc['plan_sha256']=digest(next_plan)
        receipt.write_text(json.dumps(doc))
        with patch.object(control,'scope_guard'):
            with self.assertRaises(ValueError):control.authorize(next_plan,str(receipt),control.sha(receipt))
            doc['new_independent_experiment_authorized']=True;doc['prior_experiment_series']=control.INITIAL_SERIES
            receipt.write_text(json.dumps(doc)); self.assertTrue(control.authorize(next_plan,str(receipt),control.sha(receipt)))
        doc['experiment_series']=control.INITIAL_SERIES;doc['plan_sha256']=digest(self.plan)
        doc['expires_utc']='2000-01-01T00:00:00+00:00';receipt.write_text(json.dumps(doc))
        with patch.object(control,'scope_guard'):
            with self.assertRaises(ValueError):control.authorize(self.plan,str(receipt),control.sha(receipt))

    def test_real_runner_rejects_changed_approval_input(self):
        path=Path(__file__).parent/'evidence/production-closure/runner/runner-beauty-cc1a1e277faa.py'
        spec=importlib.util.spec_from_file_location('synthetic_runner_guard',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        row=self.plan['approval_files']['precision_approval']
        request={'schema':1,'job_id':'synthetic-guard','argv':['/usr/bin/python3'],
                 'cwd':str(self.base),'timeout_seconds':30,'env':{},
                 'inputs':[{'path':row['path'],'sha256':row['sha256']}]}
        with patch.object(module,'verify_namespace'):
            module.verify(request)
            Path(row['path']).write_text('SYNTHETIC_DRIFT')
            with self.assertRaises(AssertionError):module.verify(request)

    def test_registry_blocks_root_task_jobroot_and_missing_ledger(self):
        control.experiment_binding(self.plan,seal=True)
        control.experiment_binding(self.plan)
        for key,value in [('root',str(self.base/'root2')),('task_id','other'),('job_root',str(self.base/'otherjobs'))]:
            changed=copy.deepcopy(self.plan);changed[key]=value
            with self.assertRaises(ValueError):control.experiment_binding(changed,seal=True)
        (Path(self.plan['root'])/'state.json').unlink()
        with self.assertRaisesRegex(ValueError,'state missing'):control.experiment_binding(self.plan,seal=True)

if __name__=='__main__': unittest.main(verbosity=2)
