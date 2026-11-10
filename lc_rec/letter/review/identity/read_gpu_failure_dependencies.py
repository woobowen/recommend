"""Read-only exact dependency/source and stopped-budget evidence; no GPU imports."""
import ast,datetime,hashlib,json,pathlib,socket
root=pathlib.Path("/data/bwwu/envs/letter-20261110/lib/python3.10/site-packages")
out={"observed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"host":socket.gethostname(),"sources":{}}
for rel,classes in [("peft/tuners/lora.py",{"Linear8bitLt"}),("bitsandbytes/autograd/_functions.py",{"MatMul8bitLt"})]:
    path=root/rel;raw=path.read_bytes();text=raw.decode();tree=ast.parse(text)
    methods={}
    for node in ast.walk(tree):
        if isinstance(node,ast.ClassDef) and node.name in classes:
            for child in node.body:
                if isinstance(child,ast.FunctionDef) and child.name in ("forward","backward"):
                    methods[node.name+"."+child.name]={"line":child.lineno,"source":ast.get_source_segment(text,child)}
    out["sources"][rel]={"path":str(path),"resolved_path":str(path.resolve()),"sha256":hashlib.sha256(raw).hexdigest(),"methods":methods}
path=pathlib.Path("/home/bwwu/lc-rec/letter-20261110/gpu-budget.json")
out["gpu_budget"]={"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"value":json.loads(path.read_text())}
print(json.dumps(out))

