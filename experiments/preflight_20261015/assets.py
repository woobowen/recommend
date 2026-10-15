"""Verify/upload only the three small, provenance-listed stage assets."""
import json
from pathlib import Path
import sys

from experiments.beauty_extensions.common import REPO, sha256, write_json
from .deploy import SUPPORT, TRANSPORT_SUPPORT


def main():
    sys.path.insert(0,str(TRANSPORT_SUPPORT/'scripts'))
    from transport import Transport
    t=Transport('new-server');t.identity()
    sources=json.loads((Path(__file__).with_name('asset-sources.json')).read_text())
    for item in sources['files']:
        p=Path(item['local_path'])
        assert p.stat().st_size==item['bytes'] and sha256(p)==item['sha256']
        assert not p.read_bytes().startswith(b'version https://git-lfs')
        exists=json.loads(t.python('import pathlib,json; print(json.dumps(pathlib.Path('+repr(item['remote_path'])+').exists()))'))
        if not exists:t.put(p,item['remote_path'])
    records=[{'path':v['remote_path'],'sha256':v['sha256'],'bytes':v['bytes']} for v in sources['files']]
    records.append({'path':'/data/bwwu/sid-extensions-assets/563dc50b1c7e/experiments/sports_rkmeans_qwen_20261010_093035/semantic_ids.pt',
        'sha256':'89b2c2bbb94eda26ef0e147404fbff29e6ec8246fbdc754c5d885f313f2de293','bytes':588597})
    code='''import pathlib,json,hashlib,os
records=RECORDS
for r in records:
 p=pathlib.Path(r['path']);assert p.is_file() and os.access(p,os.R_OK)
 assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
 assert not p.read_bytes().startswith(b'version https://git-lfs')
 r['resolved']=str(p.resolve())
splits={}
for domain in ['beauty','toys','sports']:
 base=pathlib.Path('/data/bwwu/sid-extensions-assets/563dc50b1c7e/original_assets/data/amazon_data')/domain
 splits[domain]={}
 for split in ['training','evaluation','testing']:
  p=base/split; files=sorted(p.glob('*.tfrecord.gz'));assert files and all(os.access(v,os.R_OK) for v in files)
  splits[domain][split]={'files':len(files),'resolved':str(p.resolve()),'first':str(files[0]),'last':str(files[-1]),'total_bytes':sum(v.stat().st_size for v in files)}
print(json.dumps({'verified_assets':records,'split_metadata':splits,'source_assets_modified':False}))
'''.replace('RECORDS',repr(records))
    result=json.loads(t.python(code));write_json(SUPPORT/'asset-presence.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
