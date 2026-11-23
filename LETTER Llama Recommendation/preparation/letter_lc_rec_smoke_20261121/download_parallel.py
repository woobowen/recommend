"""Reuse the previously verified resumable range-transfer primitive."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import itertools

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('range_transfer',
    ROOT / 'codex/llm2vec_sid_hierarchy_20261113/download_models.py')
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)
transfer.ROOT = ROOT
transfer.RELAY = 'http://127.0.0.1:18779'
original_get = transfer.requests.get
request_counter = itertools.count()
def distributed_get(url, **kwargs):
    if url.startswith(transfer.RELAY):
        url = url.replace(transfer.RELAY, f'http://127.0.0.1:{18779 + next(request_counter) % 4}', 1)
    return original_get(url, **kwargs)
transfer.requests.get = distributed_get
audit = json.loads((HERE / 'expected_source_assets.json').read_text())
entries = [(f"models/huggyllama-llama-7b/{f['name']}", {'Size':f['bytes'],'Sha256':f['sha256']})
           for f in audit['model_files']]
assert all(meta['Sha256'] for _,meta in entries)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(transfer.download, sorted(entries,key=lambda x:x[1]['Size'])))
(HERE / 'model_download_verified.json').write_text(json.dumps({
    'complete':True,'files':results,'total_bytes':sum(r['bytes'] for r in results),
    'time_unix':time.time()},indent=2))
subprocess.run([sys.executable,str(HERE/'prepare_assets.py'),'--weights',
    '--model-dir',str(ROOT/'models/huggyllama-llama-7b'),
    '--relay-base',transfer.RELAY],check=True)
