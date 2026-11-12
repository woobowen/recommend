"""Opt-in observation only. No RNG use, parameter mutation, or kernel changes."""
import hashlib
import json
from pathlib import Path
import enum

def tensor(t):
    import torch
    x=t.detach().to('cpu').contiguous()
    raw=x.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {'shape':list(x.shape),'dtype':str(x.dtype),'sha256':hashlib.sha256(raw).hexdigest(),
            'elements':x.numel()}

def tree(x):
    import torch
    import numpy as np
    if isinstance(x,torch.Tensor):return tensor(x)
    if isinstance(x,(torch.dtype,torch.device)):return str(x)
    if isinstance(x,np.ndarray):return {'ndarray':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.tobytes()).hexdigest()}
    if isinstance(x,dict):return {str(k):tree(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [tree(v) for v in x]
    if isinstance(x,enum.Enum):return {'enum':type(x).__module__+'.'+type(x).__qualname__,'name':x.name,'value':x.value}
    if isinstance(x,type) and issubclass(x,enum.Enum):return {'enum_class':x.__module__+'.'+x.__qualname__,'members':{k:v.value for k,v in x.__members__.items()}}
    if x is None or isinstance(x,(int,float,str,bool)):return x
    if type(x).__module__.startswith('deepspeed.') and hasattr(x,'__dict__'):return {'class':type(x).__module__+'.'+type(x).__qualname__,'fields':tree(vars(x))}
    return {'UNVERIFIED_STATE':type(x).__module__+'.'+type(x).__qualname__}

class Probe:
    def __init__(self,config,out,rank,event):
        self.config=json.loads(Path(config).read_text());self.out=Path(out);self.rank=rank;self.event=event
        self.step=-1;self.micro=-1;self.handles=[]
    def save(self,name,value):
        p=self.out/(name+'.rank%d.json'%self.rank)
        assert not p.exists(),str(p)
        p.write_text(json.dumps(value,indent=2)+'\n')
    def state(self,label,model,engine,scheduler,rng):
        import bitsandbytes as bnb
        modules={}
        for n,m in model.named_modules():
            if isinstance(m,bnb.nn.Linear8bitLt):
                modules[n]={'weight_attrs':{k:tree(getattr(m.weight,k,None)) for k in ['CB','SCB','has_fp16_weights']},
                            'state':tree(vars(m.state)),
                            'class':type(m).__module__+'.'+type(m).__qualname__,
                            'active_adapter':getattr(m,'active_adapter',None),'disable_adapters':getattr(m,'disable_adapters',None),
                            'aliases':{'weight_data_CB': m.state.CB is not None and m.weight.data.data_ptr()==m.state.CB.data_ptr()}}
        params={n:{'dtype':str(v.dtype),'requires_grad':v.requires_grad,'shape':list(v.shape)} for n,v in model.named_parameters()}
        runtime={k:tree(getattr(engine,k,None)) for k in ['global_steps','micro_steps','skipped_steps','global_samples','_is_gradient_accumulation_boundary']}
        runtime['zero']={k:tree(getattr(engine.optimizer,k,None)) for k in ['micro_step_id','is_gradient_accumulation_boundary','partition_gradients','gradient_accumulation_steps','dtype','communication_data_type','overlap_comm','contiguous_gradients','round_robin_gradients','averaged_gradients','params_already_reduced']}
        self.save(label,{'label':label,'rank':self.rank,'model':tree(model.state_dict()),'optimizer':tree(engine.optimizer.state_dict()),
                         'scheduler':tree(scheduler.state_dict()),'rng':tree(rng),'bnb':modules,'parameters':params,
                         'gradients':{n:tree(v.grad) for n,v in model.named_parameters() if v.requires_grad},'runtime':runtime})
        self.event('diagnosis_state',label=label,step=engine.global_steps)
    def install(self,model,engine,scheduler,rng_capture):
        import torch
        self.model=model;self.engine=engine;self.scheduler=scheduler;self.rng_capture=rng_capture
        targets=['layers.0.self_attn.q_proj','layers.31.self_attn.q_proj','lm_head.modules_to_save.default']
        for n,m in model.named_modules():
            if any(n.endswith(x) for x in targets):
                def hook(mod,args,output,name=n):
                    if self.step in [2,3]:
                        # Full selected layer outputs, no all-model activation dump.
                        self.event('diagnosis_forward',step=self.step,micro=self.micro,module=name,grad_enabled=torch.is_grad_enabled(),output=tensor(output))
                self.handles.append(m.register_forward_hook(hook))
        for n,v in model.named_parameters():
            if ('layers.0.self_attn.q_proj.lora_' in n or 'layers.31.self_attn.q_proj.lora_' in n):
                def gh(g,name=n):
                    if self.step in [2,3]:self.event('diagnosis_parameter_gradient',step=self.step,micro=self.micro,parameter=name,gradient=tensor(g),norm=float(g.float().norm()),max_abs=float(g.abs().max()))
                    return g
                self.handles.append(v.register_hook(gh))
        original=engine.optimizer.step
        def observed_step(*args,**kwargs):
            s=engine.global_steps
            if s in [2,3]:
                self.save('S4-step%d'%s,{'rank':self.rank,'step_before_update':s,
                    'averaged_gradients':tree(engine.optimizer.averaged_gradients),
                    'master':tree(engine.optimizer.single_partition_of_fp32_groups),
                    'optimizer':tree(engine.optimizer.optimizer.state_dict())})
            result=original(*args,**kwargs)
            if s in [2,3]:
                self.save('S5-step%d'%s,{'rank':self.rank,'step_before_update':s,
                    'trainable':tree({n:v for n,v in model.named_parameters() if v.requires_grad}),
                    'optimizer':tree(engine.optimizer.state_dict())})
            return result
        engine.optimizer.step=observed_step
    def batch(self,step,micro):
        self.step=step;self.micro=micro
        if step==2 and micro==9:
            self.state('before_first_forward',self.model,self.engine,self.scheduler,self.rng_capture())
