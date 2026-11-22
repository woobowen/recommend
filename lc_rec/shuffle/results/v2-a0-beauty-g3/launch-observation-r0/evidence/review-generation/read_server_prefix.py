"""Read a single six-line byte prefix and its small model-boundary evidence."""
import argparse
import base64
import json
from pathlib import Path
import re

from lc_rec.shuffle_v2 import g3_control as gc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', default='r0')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert re.fullmatch(r'r[0-9]+', args.attempt)
    folder = f'{gc.ROOT}/gpu-formal-shuffle2-{args.attempt}'
    target = gc.connection()[0]
    packet = json.loads(gc.control.remote_python(target, f"""
import base64, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
root=Path({folder!r})
start=datetime.now(timezone.utc).isoformat()
prediction=root/'predictions.rank0.jsonl'
with prediction.open('rb') as f:
 st=os.fstat(f.fileno())
 rows=[f.readline(1024*1024) for _ in range(6)]
 assert all(line.endswith(b'\\n') and isinstance(json.loads(line),dict) for line in rows)
 raw=b''.join(rows)
 assert 0<len(raw)<=st.st_size
 rank=(root/'rank0.json').read_bytes()
 binding=(root/'v2-binding.rank0.json').read_bytes()
 derived=(root/'derived.rank0.py').read_bytes()
 files={{'first-six.jsonl':raw,'rank0.json':rank,'v2-binding.rank0.json':binding,'derived.rank0.py':derived}}
 out={{'started_utc':start,'ended_utc':datetime.now(timezone.utc).isoformat(),'prediction_path':str(prediction),'file_bytes_at_open':st.st_size,'file_inode':st.st_ino,'first_six_end_offset':len(raw),'files':{{name:{{'sha256':hashlib.sha256(data).hexdigest(),'content':base64.b64encode(data).decode()}} for name,data in files.items()}}}}
 print(json.dumps(out))
""", timeout=30))
    args.output.mkdir()
    for name, value in packet['files'].items():
        (args.output / name).write_bytes(base64.b64decode(value.pop('content'), validate=True))
    (args.output / 'direct-prefix-readback.json').write_text(json.dumps(packet, indent=2) + '\n')
    print(json.dumps(packet, indent=2))


if __name__ == '__main__':
    main()
