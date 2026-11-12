"""CPU-only acceptance of the actual ReplayLoader and checkpoint rotation."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'DS_ACCELERATOR' not in os.environ
    import torch
    from transformers import Trainer
    from formal_train import write,sha,checkpoint_required_files
    from formal_attention import INTERFACE_SHA,FASTCHAT_SHA,EXTENSION_SHA
    import importlib.util
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    assert not torch.cuda.is_initialized()
    source=Path(__file__).with_name('formal_train.py').read_text();tree=ast.parse(source)
    cls=next(x for x in ast.walk(tree) if isinstance(x,ast.ClassDef) and x.name=='AuditTrainer')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_rotate_checkpoints')
    derived=ast.ClassDef(name='Rotation',bases=[ast.Name(id='Trainer',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])
    ns={'Trainer':Trainer}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[derived],type_ignores=[])),'actual_rotation','exec'),ns)
    actual=ns['Rotation'].__new__(ns['Rotation']);base=output/'rotation_fixture';base.mkdir(exist_ok=False)
    for step in [2060,4120,6180]:(base/('checkpoint-%d'%step)).mkdir()
    actual.args=SimpleNamespace(save_total_limit=2);actual.state=SimpleNamespace(best_model_checkpoint=str(base/'checkpoint-2060'))
    actual._writing_checkpoint=True;actual._rotate_checkpoints(output_dir=str(base))
    assert len(list(base.glob('checkpoint-*')))==3
    actual._writing_checkpoint=False;actual._rotate_checkpoints(output_dir=str(base))
    assert {x.name for x in base.glob('checkpoint-*')}=={'checkpoint-2060','checkpoint-6180'}
    required=checkpoint_required_files(2060);assert len(required)==17
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'config/formal_instruments_seed42.json').read_text())
    author=json.loads((root/'config/author_instruments.json').read_text())
    for k,v in author.items():
        if k not in ['base_model','output_dir','resume_from_checkpoint','deepspeed','data_path','gradient_accumulation_steps']:assert config[k]==v,k
    from formal_evaluate import catalog_and_users
    _,catalog,users,assets=catalog_and_users(root/'vendor/LETTER/data')
    fa=Path(importlib.util.find_spec('flash_attn').origin).parent
    fc=Path(importlib.util.find_spec('fastchat').origin).parent
    assert sha(fa/'flash_attn_interface.py')==INTERFACE_SHA
    assert sha(fc/'train/llama2_flash_attn_monkey_patch.py')==FASTCHAT_SHA
    extension=Path(importlib.util.find_spec('flash_attn_2_cuda').origin)
    assert sha(extension)==EXTENSION_SHA
    result={'status':'PASS','cpu_only':True,'torch_cuda_initialized':torch.cuda.is_initialized(),
            'run_source_sha256':sha(Path(__file__).with_name('formal_train.py')),
            'rotation':{'actual_method_ast_sha256':hashlib.sha256(ast.dump(method).encode()).hexdigest(),
                        'deferred_until_complete':True,'keeps_old_best_and_new_latest':True,'effective_limit':2},
            'assets':assets,'users':len(users),'catalog_sids':len(catalog),
            'checkpoint_required_files':sorted(required),'config':config,
            'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','accelerate','deepspeed','peft','bitsandbytes','fschat','flash-attn']},
            'extension':{'path':str(extension),'sha256':EXTENSION_SHA}}
    assert not torch.cuda.is_initialized();write(output/'cpu_acceptance.json',result)
    from formal_bnb_reload import cpu_regression
    write(output/'bnb_hot_reload_cpu.json',cpu_regression())
    scripts=Path(__file__).parent
    for script,extra in [('test_formal_evaluate.py',[]),('test_formal_replay.py',['--output',str(output/'replay.json')])]:
        cp=subprocess.run([sys.executable,str(scripts/script),*extra],capture_output=True,text=True)
        write(output/(script+'.json'),{'argv':cp.args,'exit_code':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr})
        assert cp.returncode==0,cp.stderr
    print(json.dumps({'status':'PASS','output':str(output)}))


if __name__=='__main__':main()
