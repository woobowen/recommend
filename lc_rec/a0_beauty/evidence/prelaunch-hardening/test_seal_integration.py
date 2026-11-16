"""CPU-only isolated test: real plan builder, server assets and pending review.

Only the synthetic-root rejection is patched while preparing; the production
validation without that patch must reject this fixture. No PASS review or user
launch approval is created, and this script never invokes start/step/run.
"""
import copy, json, os, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
release=Path('/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/releases/6e1b211d9c37f358c43b1540b40b17f6703150d2')
sys.path.insert(0,str(release/'lc_rec/a0_beauty'))
import production_control as control
import prepare_production_plan as builder
from formal_control import read

class SealTest(unittest.TestCase):
    def test_pending_review_can_prepare_only_in_isolated_fixture(self):
        base=Path('/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/formal-candidate-5b2917d')
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_SEAL_') as temp:
            root=Path(temp)/'SYNTHETIC_SEAL_ROOT';review=Path(temp)/'SYNTHETIC_PENDING_REVIEW.json'
            review.write_text(json.dumps({'scope':'SYNTHETIC_TEST','status':'PENDING'}))
            argv=['prepare_production_plan.py','--deployment',str(release/'RUN_VERSION.json'),
              '--root',str(root),'--data-root','/home/bwwu/grid-audits/sid-extensions-p0/transfers/letter-a0-beauty-author-8d0154e28de3',
              '--source-root','/home/bwwu/grid-audits/sid-extensions-p0/transfers/beauty-infer-63f13def364d/release',
              '--model','/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549',
              '--python','/data/bwwu/envs/letter-20261110/bin/python','--runner',str(base/'runner-beauty.py'),
              '--precision-approval','/home/bwwu/lc-rec/letter-a0-normal-det-20261112/releases/4c6a83d9ba42dc2fe737792759e410ef1874da76/lc_rec/letter/evidence/formal_20261111/precision_approval.json',
              '--prelaunch-review',str(review),'--budget',str(base/'budget-candidate.json'),
              '--task-id','synthetic-fixture','--gpu-uuids','GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae','GPU-59cac3bd-3645-668b-79c8-839e0d878893']
            with patch.object(sys,'argv',argv),patch.object(control,'scope_guard'):
                builder.main()
            plan=read(root/'plan.json')
            with self.assertRaises(ValueError):control.validate_plan(plan)
            with patch.object(control,'scope_guard'):
                invalid=copy.deepcopy(plan);invalid['gpu_uuids']=[]
                with self.assertRaisesRegex(ValueError,'two fixed physical GPU'):control.validate_plan(invalid)
                invalid=copy.deepcopy(plan);invalid['config_sha256']='0'*64
                with self.assertRaisesRegex(ValueError,'configuration SHA'):control.validate_plan(invalid)
            with self.assertRaises(PermissionError):control.authorize(plan,None,None)
            with self.assertRaises((KeyError,ValueError)):control.approval_documents(plan,require_pass=True)
            self.assertFalse((root/'state.json').exists())

if __name__=='__main__': unittest.main(verbosity=2)
