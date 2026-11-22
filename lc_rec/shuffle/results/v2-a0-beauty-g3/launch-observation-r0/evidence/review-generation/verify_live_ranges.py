"""Hash the already-observed byte ranges once on new-server, without a status query."""
import argparse
import json
from pathlib import Path

from lc_rec.shuffle_v2 import g3_control as gc

SOURCE = """
from datetime import datetime,timezone
from hashlib import sha256
import json,os
from pathlib import Path

start=datetime.now(timezone.utc).isoformat()
results=[]
for label,point in evidence.items():
 proof={}
 for name,t in point.items():
  path=Path(t['path'])
  wanted=t['end']-t['start']
  assert wanted>=0
  with path.open('rb') as f:
   identity=os.fstat(f.fileno())
   assert (identity.st_dev,identity.st_ino)==(t['device'],t['inode'])
   assert identity.st_size>=t['end']
   f.seek(t['start'])
   digest=sha256()
   remaining=wanted
   count=0
   while remaining:
    chunk=f.read(min(remaining,1024**2))
    assert chunk
    remaining-=len(chunk)
    digest.update(chunk)
    count+=chunk.count(b'\\n')
   assert f.tell()==t['end']
   assert digest.hexdigest()==t['sha256'], (label,name)
   if t['rows'] is not None:
    assert count==t['rows'], (label,name,count)
   proof[name]={'start':t['start'],'end':t['end'],'bytes':wanted,'sha256':digest.hexdigest(),
                'complete_rows':count if t['rows'] is not None else None,
                'device':identity.st_dev,'inode':identity.st_ino}
 results.append({'observation':label,'checked_byte_ranges':proof})
print(json.dumps({'status':'PASS_LIVE_CAPTURED_BYTE_RANGES_NON_AUTHOR','started_utc':start,
                  'finished_utc':datetime.now(timezone.utc).isoformat(),
                  'evidence':results},allow_nan=False))
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    args = p.parse_args()
    evidence = {}
    for minute in range(0,31,5):
        label = f'T{minute:02d}'
        point = json.loads((args.root/'observations'/(label+'.json')).read_bytes())
        assert point['label'] == label
        pred = point['prediction']
        ranges = {}
        if 'delta_offset_start' in pred:
            ranges['prediction'] = {'start':pred['delta_offset_start'],'end':pred['complete_bytes'],
                'inode':pred['inode'],'device':pred['device'],'path':pred['path'],
                'sha256':pred['delta_complete_sha256'],'rows':pred['delta_users']}
        for name, log in point['logs'].items():
            if log is not None:
                ranges[name] = {'start':log['offset_start'],'end':log['offset_end'],
                    'inode':log['inode'],'device':log['device'],'path':log['path'],
                    'sha256':log['excerpt_sha256'],'rows':None}
        evidence[label] = ranges
    server = gc.connection()[0]
    report = json.loads(gc.control.remote_python(server, 'evidence = ' + repr(evidence) + '\n' + SOURCE, timeout=45))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
