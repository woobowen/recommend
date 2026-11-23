"""Exercise diagnostic callback/config setup without any optimization."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',TF_CPP_MIN_LOG_LEVEL='3',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='2')
import sys,json,runpy
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.utils.custom_hydra_resolvers
import lightning
import torch
torch.set_num_threads(4)
OUT=ROOT/'experiments/yelp_diagnostic_callback_check'

class Checked(Exception):
    pass

def check_fit(trainer,model,datamodule=None,ckpt_path=None):
    from codex.yelp_diagnostics.extended_monitor import ExtendedMonitor
    callback=next(c for c in trainer.callbacks if isinstance(c,ExtendedMonitor))
    state=torch.get_rng_state().clone()
    callback.on_validation_end(SimpleNamespace(sanity_checking=False,global_step=0,
        callback_metrics={'val/recall@5':torch.tensor(0.)}),model)
    assert torch.equal(state,torch.get_rng_state())
    assert json.loads((OUT/'latest_convergence.json').read_text())['validation_users']==100
    (OUT/'check_passed.json').write_text(json.dumps(dict(passed=True,optimizer_steps=0,
        validation_callback_exercised=True,torch_rng_preserved=True,config_diff_verified=True)))
    raise Checked()

lightning.Trainer.fit=check_fit
sys.argv=['run_extended_seed43.py','--dataset','yelp_aligned_v1','--variant','shuffle','--gpu','2',
    '--seed','43','--output',str(OUT),'--sample-dir',str(ROOT/'experiments/sid_order_yelp_5seed_v1/fixed_test_users')]
try:
    runpy.run_path(str(ROOT/'codex/yelp_diagnostics/run_extended_seed43.py'),run_name='__main__')
except Checked:
    print('Diagnostic callback and config check passed; zero optimizer steps.')
