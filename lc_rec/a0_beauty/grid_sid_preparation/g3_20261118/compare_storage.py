#!/usr/bin/env python3
"""Cross-check the two verified PyTorch ZIP storages without unpickling."""
import hashlib, json, pathlib, pickletools, struct, zipfile
assets={'G3':('yjc_semantic_ids.pt','d41adcfb22a9a47d5e0424c068f4c74f33cc9fdbaf5db08540fbfa0e3d720e18'),'merged':('yjc_merged_predictions_tensor.pt','d5241c43708cb9e5ce0c1e812b965a0b5f92e38d94aead8246bcebeea62e385f')}
raws={}; metadata={}
for name,(fn,expected) in assets.items():
 b=pathlib.Path(fn).read_bytes(); assert hashlib.sha256(b).hexdigest()==expected
 with zipfile.ZipFile(fn) as z:
  paths=z.namelist(); p=next(s for s in paths if s.endswith('/data.pkl')); data=z.read(p)
  ops=[(op.name,arg) for op,arg,_ in pickletools.genops(data)]
  assert ('GLOBAL','torch LongStorage') in ops and ('GLOBAL','torch._utils _rebuild_tensor_v2') in ops
  assert ops==metadata.get('ops',ops); metadata['ops']=ops
  assert z.read(next(s for s in paths if s.endswith('/byteorder')))==b'little'
  raw=z.read(next(s for s in paths if s.endswith('/data/0'))); assert len(raw)==4*12101*8
  raws[name]=raw
# The identical, disassembled metadata describes shape (4,12101), strides (1,4), offset 0.
a,b=raws.values(); rows_a=list(struct.iter_unpack('<4q',a)); rows_b=list(struct.iter_unpack('<4q',b)); assert len(rows_a)==12101
result={'identical_pickle_metadata':True,'identical_storage_bytes':a==b,'storage_sha256':hashlib.sha256(a).hexdigest(),'items_equal':sum(x==y for x,y in zip(rows_a,rows_b)),'per_position_differences':[sum(x[i]!=y[i] for x,y in zip(rows_a,rows_b)) for i in range(4)],'storage_bytes':len(a),'layout_basis':'pickletools metadata shape [4,12101], stride [1,4], offset 0; little endian int64'}
print(json.dumps(result,indent=2))
