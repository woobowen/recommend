"""Read-only independent server B asset and tokenizer implementation evidence."""
import ast, collections, datetime, hashlib, json, pathlib, socket
base=pathlib.Path("/data/bwwu/lc-rec/assets/data/Instruments")
result={"observed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"host":socket.gethostname(),"assets":{}}
parsed={}
def canon(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
for kind in ("inter","item","index"):
    p=base/("Instruments."+kind+".json")
    b=p.read_bytes(); x=json.loads(b); parsed[kind]=x
    result["assets"][kind]={"path":str(p),"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest(),"canonical_sha256":canon(x),"entries":len(x)}
inters,items,idx=[parsed[k] for k in ("inter","item","index")]
groups=collections.defaultdict(list)
for iid,sid in idx.items(): groups[tuple(sid)].append(iid)
used={str(i) for seq in inters.values() for i in seq}
result["semantics"]={"users":len(inters),"metadata_items":len(items),"index_items":len(idx),"used_items":len(used),"all_axes_equal":used==set(items)==set(idx),"interactions":sum(map(len,inters.values())),"train_rows":sum(len(x)-3 for x in inters.values()),"unique_sid":len(groups),"sid_length_histogram":dict(collections.Counter(map(len,idx.values()))),"collision_groups":sum(len(g)>1 for g in groups.values()),"colliding_items":sum(len(g) for g in groups.values() if len(g)>1),"extra_collisions":sum(len(g)-1 for g in groups.values() if len(g)>1),"collision_item_groups":sorted(sorted(g) for g in groups.values() if len(g)>1),"unique_sid_tokens":len({t for sid in idx.values() for t in sid})}
result["split_labels"]={}
for mode in ("train","valid","test"):
    labels=[]; collided=0
    for user,seq in inters.items():
        positions=range(1,len(seq)-2) if mode=="train" else [len(seq)-(2 if mode=="valid" else 1)]
        for position in positions:
            target=str(seq[position]);history=list(map(str,seq[max(0,position-20):position]))
            labels.append([user,position,target])
            collided+=len(groups[tuple(idx[target])])>1
    result["split_labels"][mode]={"rows":len(labels),"labels_sha256":canon(labels),"colliding_target_rows":collided}
p=pathlib.Path("/data/bwwu/envs/lc-rec-offload-20261108/lib/python3.10/site-packages/transformers/tokenization_utils.py")
text=p.read_text(); tree=ast.parse(text)
result["tokenizer_parent_implementation"]={"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
for node in ast.walk(tree):
    if isinstance(node,ast.ClassDef) and node.name=="PreTrainedTokenizer":
        for child in node.body:
            if isinstance(child,ast.FunctionDef) and child.name in ("tokenize","_add_tokens"):
                result["tokenizer_parent_implementation"][child.name]={"start_line":child.lineno,"source":ast.get_source_segment(text,child)}
job=pathlib.Path("/home/bwwu/grid-audits/sid-extensions-p0/jobs/letter-20261110-data-r1/request.json")
raw=job.read_bytes()
result["remote_request"]={"path":str(job),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"canonical_sha256":canon(json.loads(raw))}
print(json.dumps(result,ensure_ascii=False))

