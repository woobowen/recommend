#!/usr/bin/env bash
set -euo pipefail
repo="${1:-/home/addaswsw/recommend-worktrees/shuffle-v2}"
revision="${2:-working-copy}"
here="$(dirname "$(realpath "$0")")"

python3 - "$repo" "$revision" "$here/native_eager_probe.py" <<'PY' | /mnt/c/Windows/System32/OpenSSH/ssh.exe -T -o BatchMode=yes -o StrictHostKeyChecking=yes new-server 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3 /data/bwwu/envs/letter-20261110/bin/python -B -'
from pathlib import Path
import base64,hashlib,subprocess,sys
repo=Path(sys.argv[1])
rev=sys.argv[2]
prefix='lc_rec.beauty_history_attention'
print('import base64,sys,types')
print('source_shas = {}')
for name in ('lc_rec',prefix):
    print(f"package=types.ModuleType({name!r}); package.__path__=[]; sys.modules[{name!r}]=package")
for name in ('attention','inference_attention'):
    path='lc_rec/beauty_history_attention/'+name+'.py'
    if rev == 'working-copy':
        content=(repo/path).read_bytes()
    else:
        content=subprocess.check_output(['git','-C',str(repo),'show',rev+':'+path])
    key=prefix+'.'+name
    print(f'subject=types.ModuleType({key!r}); sys.modules[{key!r}]=subject; '
          f'subject.__package__={prefix!r}; subject.__file__={path!r}')
    print(f'exec(compile(base64.b64decode({base64.b64encode(content).decode()!r}),{path!r},"exec"),subject.__dict__)')
    print(f'source_shas[{name!r}]={hashlib.sha256(content).hexdigest()!r}')
print(Path(sys.argv[3]).read_text())
PY
