"""Independent bounded numeric readers, ranking metrics and attention paths.

No research producer, torch, checkpoint object, evaluator, GPU or model imports.
"""
from pathlib import Path
from collections import OrderedDict
import hashlib,io,json,math,pickle,zipfile
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def ident(p):
    p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def read_rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]

def read_sid_tensor(p):
    """Restricted numeric decoder for the fixed small LongStorage SID artifact."""
    with zipfile.ZipFile(p) as archive:
        names=archive.namelist();prefix=next(n[:-len('data.pkl')] for n in names if n.endswith('/data.pkl'))
        assert archive.read(prefix+'byteorder')==b'little'
        class LongStorage:pass
        def rebuild(storage,offset,size,stride,requires_grad,hooks):
            assert not requires_grad and not hooks and offset==0
            assert tuple(size)==(4,18357) and tuple(stride)==(1,4)
            assert storage.dtype==np.dtype('<i8') and storage.size==73428
            return np.ndarray(tuple(size),dtype='<i8',buffer=storage,strides=tuple(x*8 for x in stride)).copy()
        class Restricted(pickle.Unpickler):
            def find_class(self,module,name):
                allowed={('torch._utils','_rebuild_tensor_v2'):rebuild,('torch','LongStorage'):LongStorage,('collections','OrderedDict'):OrderedDict}
                assert (module,name) in allowed,(module,name)
                return allowed[(module,name)]
            def persistent_load(self,pid):
                assert pid==('storage',LongStorage,'0','cpu',73428),pid
                raw=archive.read(prefix+'data/0');assert len(raw)==73428*8
                return np.frombuffer(raw,dtype='<i8')
        result=Restricted(io.BytesIO(archive.read(prefix+'data.pkl'))).load()
    assert result.shape==(4,18357) and result.dtype==np.int64
    return result

def read_bundle(p):
    p=Path(p);doc=load(p);seen=set()
    assert doc['schema']=='grid-beauty-cross-attention-v1' and doc['arrays_file']=='arrays.npz'
    assert sha(p.with_name('arrays.npz'))==doc['arrays_sha256']
    with np.load(p.with_name('arrays.npz'),allow_pickle=False) as arrays:
        def decode(x):
            if isinstance(x,dict) and '__array__' in x:
                a=arrays[x['__array__']].copy();seen.add(x['__array__'])
                assert list(a.shape)==x['shape'] and str(a.dtype)==x['dtype'];return a
            if isinstance(x,dict):return {k:decode(v) for k,v in x.items()}
            if isinstance(x,list):return [decode(v) for v in x]
            return x
        result=decode(doc['bundle']);assert seen==set(arrays.files)
    return result

def ranking(row):
    """Exact SID match with unmodified scores; explicit tie ambiguity guard."""
    sids=[tuple(v) for v in row['sids']];target=tuple(row['target']);scores=np.array(row['scores'],dtype=np.float32)
    assert len(sids)==10 and scores.shape==(10,) and np.isfinite(scores).all()
    assert np.all(scores[:-1]>=scores[1:])
    ordered=[sids[i] for i in np.argsort(-scores,kind='stable')]
    rank=ordered.index(target)+1 if target in ordered else 0
    target_tied=False
    if rank:target_tied=int(np.count_nonzero(scores==scores[rank-1]))>1
    assert not target_tied,'Relevant score tie needs explicit original topk resolution; no fabricated stable tie order.'
    metrics={}
    for k in [5,10]:
        metrics[f'recall@{k}']=float(0<rank<=k)
        metrics[f'ndcg@{k}']=1/math.log2(rank+1) if 0<rank<=k else 0.0
    return {'target':target,'ordered':ordered,'rank':rank,'metrics':metrics,
            'score_tied':len(set(scores.tolist()))<10}

def pair_metrics(x,y):
    norm=lambda z:math.sqrt(math.fsum(float(v)*float(v) for v in z))
    dot=lambda a,b:math.fsum(float(v)*float(w) for v,w in zip(a,b))
    cosine=dot(x,y)/(norm(x)*norm(y));xc=x-x.mean();yc=y-y.mean();den=norm(xc)*norm(yc)
    p=x/math.fsum(x);q=y/math.fsum(y);mid=(p+q)/2
    def kl(z):return math.fsum(float(v)*math.log(float(v/mid[i])) for i,v in enumerate(z) if v>0)
    return {'cosine':cosine,'pearson':dot(xc,yc)/den if den else None,
        'total_variation':math.fsum(abs(p-q))/2,'jensen_shannon_nats':max(0.0,(kl(p)+kl(q))/2)}

def independent_paths(steps):
    assert len(steps)==4
    previous=None;parents=[];head_sum_error=0.0
    for h,step in enumerate(steps):
        ids=step['generated_ids'];heads=step['cross_heads']
        assert ids.shape==(8,10,h+1) and heads.shape==(8,1 if h==0 else 10,6,150)
        assert step['hierarchy']==h and heads.dtype==np.float32 and np.isfinite(heads).all() and (heads>=0).all()
        head_sum_error=max(head_sum_error,float(np.max(np.abs(heads.sum(-1)-1))))
        if previous is None:parent=np.zeros((8,10),dtype=np.int64)
        else:
            matches=(ids[:,:,None,:-1]==previous[:,None,:,:]).all(-1)
            assert (matches.sum(-1)==1).all();parent=matches.argmax(-1)
        assert np.array_equal(parent,step['parent_indices'])
        for row in ids:assert len(set(map(tuple,row)))==10
        parents.append(parent);previous=ids
    final=np.empty((8,4,6,150),np.float32);fp=np.empty((8,4),np.int64);fc=np.empty((8,4),np.int64);tokens=np.empty((8,4),np.int64)
    for sample in range(8):
        child=0
        for h in range(3,-1,-1):
            parent=int(parents[h][sample,child]);fp[sample,h]=parent;fc[sample,h]=child
            final[sample,h]=steps[h]['cross_heads'][sample,parent]
            tokens[sample,h]=steps[h]['generated_ids'][sample,child,-1];child=parent
    assert np.array_equal(tokens,steps[-1]['generated_ids'][:,0])
    local=np.stack([s['cross_heads'][:,0] for s in steps],axis=1)
    return {'final_top1_heads':final,'local_first_heads':local,'final_top1_cross':final.mean(2),'local_first_cross':local.mean(2),
            'final_parent_indices':fp,'final_child_indices':fc,'backtracked_top1_sid':tokens},head_sum_error
