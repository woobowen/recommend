#!/usr/bin/env python3
"""Small paired bootstrap regression; no full Beauty prediction evaluation."""
import importlib.util
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('verify',root/'letter/review/final_audit_20261113/statistics_verify.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a=np.array([[0.,1.,.2],[1.,1.,.4],[0.,0.,.6],[1.,0.,.8]])
b=np.array([[1.,1.,.3],[1.,0.,.4],[0.,1.,.5],[0.,0.,.9]])
d=b-a;rng=np.random.Generator(np.random.PCG64(20261113));draw=np.empty((101,3))
for begin in range(0,101,8):
 choice=rng.integers(0,len(d),size=(min(8,101-begin),len(d)),dtype=np.int64);draw[begin:begin+len(choice)]=d[choice].mean(axis=1)
ci,reference=m.bootstrap_columns(d,repetitions=101)
assert np.max(np.abs(draw-reference))<=1e-13
assert np.max(np.abs(np.percentile(draw,[2.5,97.5],axis=0,method='linear')-ci))<=1e-13
src=(Path(__file__).parent/'paired_beauty.py').read_text();assert '24772' not in src and '22363' in src and '7430079e2b917dfa5e4bb8ead2702fb6c622fb48ffb684dd2889780f9480af7f' in src
print('PASS: small paired user bootstrap and Beauty axis constants; not full evaluation')
