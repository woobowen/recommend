"""Layer28 first-resumed-microbatch observations; original tensors/gradients returned.

CPU copies synchronize CUDA. Paired runs use the same observers, but this can
change scheduling. Installed packages, model parameters and training ops stay
unchanged. Raw FA backward results are distinct from post-unpad/RoPE gradients.
"""
import functools
import hashlib
import inspect
import json
from pathlib import Path


class Boundary:
    def __init__(self, probe):
        self.probe=probe
        self.root=probe.out/('boundary.rank%d'%probe.rank)
        self.root.mkdir(exist_ok=False)
        self.active=None
        self.native=None
        self.counts={}

    def save(self, label, value, identity, gradient=False):
        import torch
        path=self.root/(identity['phase']+'.'+label+('.grad' if gradient else '')+'.pt')
        assert not path.exists(),str(path)
        cpu=value.detach().to('cpu').contiguous().clone()
        raw=cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        row=dict(identity,label=label,gradient=gradient,shape=list(value.shape),
                 dtype=str(value.dtype),stride=list(value.stride()),layout=str(value.layout),
                 device=str(value.device),storage_offset=value.storage_offset(),
                 contiguous=value.is_contiguous(),requires_grad=value.requires_grad,
                 sha256=hashlib.sha256(raw).hexdigest(),elements=value.numel(),
                 norm=float(cpu.float().norm()),max_abs=float(cpu.float().abs().max()) if cpu.numel() else 0,
                 file=path.name)
        torch.save(cpu,path)
        path.with_suffix('.json').write_text(json.dumps(row,indent=2)+'\n')
        self.probe.event('attention_boundary',**row)

    def observe(self,label,value,identity=None):
        identity=identity or self.active
        if identity is None:return
        self.save(label,value,identity)
        if value.requires_grad:
            def hook(gradient):
                self.save(label,gradient,identity,gradient=True)
                return gradient
            value.register_hook(hook)

    def install(self, model):
        import torch
        import fastchat.train.llama2_flash_attn_monkey_patch as fc
        import flash_attn.flash_attn_interface as fa
        targets=[(n,m) for n,m in model.named_modules() if n.endswith('layers.28.self_attn')]
        assert len(targets)==1
        name,attn=targets[0]
        original=attn.forward
        @functools.wraps(original)
        def layer(*args,**kwargs):
            watched=self.probe.step==2 and self.probe.micro==9
            previous=self.active
            if watched:
                phase='recompute' if torch.is_grad_enabled() else 'primary'
                self.counts[phase]=self.counts.get(phase,0)+1
                assert self.counts[phase]==1
                self.active={'rank':self.probe.rank,'layer':28,'module':name,
                             'step':2,'micro':9,'phase':phase,'call':1}
                rotary=attn.rotary_emb
                self.probe.event('rotary_cache',**self.active,
                    max_seq_len_cached=rotary.max_seq_len_cached,
                    cos=self._fingerprint(rotary.cos_cached),sin=self._fingerprint(rotary.sin_cached))
            try:return original(*args,**kwargs)
            finally:self.active=previous
        attn.forward=layer
        for projection in ['q_proj','k_proj','v_proj','o_proj']:
            def hook(module,args,output,label=projection):
                if self.active:
                    # Shared projection input would get several hooks; only q input is observed.
                    if label=='q_proj':self.observe('projection.input',args[0])
                    if label=='o_proj':self.observe('o_proj.input',args[0])
                    self.observe(label+'.output',output)
            self.probe.handles.append(getattr(attn,projection).register_forward_hook(hook))
        rotary_original=fc.apply_rotary_pos_emb
        @functools.wraps(rotary_original)
        def rotary(q,k,cos_sin,position_ids):
            if self.active:
                for label,t in [('rope.q.pre',q),('rope.k.pre',k),('rope.cos',cos_sin[0]),
                                ('rope.sin',cos_sin[1]),('positions',position_ids)]:self.observe(label,t)
            result=rotary_original(q,k,cos_sin,position_ids)
            if self.active:
                self.observe('rope.q.post',result[0]);self.observe('rope.k.post',result[1])
            return result
        fc.apply_rotary_pos_emb=rotary
        unpad_original=fc.unpad_input
        @functools.wraps(unpad_original)
        def unpad(x,mask,*args,**kwargs):
            label='unpad.q' if x.ndim==4 else 'unpad.kv'
            self.observe(label+'.pre',x)
            if self.active:self.observe(label+'.mask',mask)
            result=unpad_original(x,mask,*args,**kwargs)
            if self.active:
                self.observe(label+'.post',result[0]);self.observe(label+'.indices',result[1])
                self.observe(label+'.cu',result[2])
                assert result[1].unique().numel()==result[1].numel(), 'duplicate scatter destinations'
            return result
        fc.unpad_input=unpad
        pad_original=fc.pad_input
        @functools.wraps(pad_original)
        def pad(x,*args,**kwargs):
            self.observe('pad.pre',x)
            result=pad_original(x,*args,**kwargs)
            self.observe('pad.post',result)
            return result
        fc.pad_input=pad
        # ctx identity survives after forward context exits and before its backward.
        cls=fa.FlashAttnVarlenKVPackedFunc
        forward_original=cls.forward;backward_original=cls.backward
        @functools.wraps(forward_original)
        def forward(ctx,*args,**kwargs):
            identity=self.active
            if identity:ctx.letter_boundary_identity=dict(identity)
            result=forward_original(ctx,*args,**kwargs)
            if identity:
                bound=inspect.signature(forward_original).bind(ctx,*args,**kwargs)
                options={k:v for k,v in bound.arguments.items() if k not in ['ctx','q','kv','cu_seqlens_q','cu_seqlens_k','alibi_slopes']}
                assert options['dropout_p']==0.0 and options['causal'] is True
                self.probe.event('fa_effective_call',**identity,function='flash_attn_varlen_kvpacked_func',options=options)
                self.observe('fa.output',result,identity)
            return result
        @functools.wraps(backward_original)
        def backward(ctx,dout,*args):
            identity=getattr(ctx,'letter_boundary_identity',None)
            if identity:
                self.native=identity
                for label,t in zip(['q','k','v','out','lse','cu_q','cu_k','rng'],ctx.saved_tensors):self.save('native.'+label,t,identity)
                self.save('native.dO',dout,identity)
                self.probe.event('fa_backward_ctx',**identity,deterministic=ctx.deterministic,
                    max_q=ctx.max_seqlen_q,max_k=ctx.max_seqlen_k,dropout_p=ctx.dropout_p,
                    softmax_scale=ctx.softmax_scale,causal=ctx.causal,window_size=ctx.window_size,
                    softcap=ctx.softcap,alibi_slopes=None if ctx.alibi_slopes is None else 'TENSOR')
            try:
                result=backward_original(ctx,dout,*args)
                if identity:
                    self.save('native.dQ',result[0],identity)
                    self.save('native.dKV',result[1],identity)
                return result
            finally:self.native=None
        cls.forward=staticmethod(forward);cls.backward=staticmethod(backward)
        native_original=fa._wrapped_flash_attn_varlen_backward
        @functools.wraps(native_original)
        def native(*args,**kwargs):
            if self.native:
                # Version-pinned ABI: argument20 is deterministic; output buffers6:9.
                assert len(args)==21
                self.probe.event('fa_native_dispatch',**self.native,deterministic=args[20],
                    callable=str(native_original),positional_arguments=len(args),
                    q_stride=list(args[1].stride()),k_stride=list(args[2].stride()),v_stride=list(args[3].stride()))
                for label,t in zip(['dO','q','k','v','out','lse'],args[:6]):self.save('dispatch.'+label,t,self.native)
            result=native_original(*args,**kwargs)
            if self.native:
                for label,t in zip(['dQ','dK','dV'],args[6:9]):self.save('dispatch.'+label,t,self.native)
            return result
        fa._wrapped_flash_attn_varlen_backward=native
        extension_original=fa.flash_attn_gpu.varlen_bwd
        @functools.wraps(extension_original)
        def extension(*args,**kwargs):
            if self.native:
                assert len(args)==24
                self.probe.event('fa_extension_dispatch',**self.native,
                    deterministic=args[21],extension_file=fa.flash_attn_gpu.__file__,
                    positional_arguments=len(args),last_dimension_strides=[t.stride(-1) for t in args[:5]])
            return extension_original(*args,**kwargs)
        fa.flash_attn_gpu.varlen_bwd=extension

    @staticmethod
    def _fingerprint(t):
        from resume_probe import tensor
        return dict(tensor(t),device=str(t.device),stride=list(t.stride()))
