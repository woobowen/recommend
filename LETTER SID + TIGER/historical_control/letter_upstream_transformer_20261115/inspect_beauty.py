"""Read the latest official LETTER Beauty downstream experiment status."""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
HERE=ROOT/'codex/letter_upstream_transformer_20261115'


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def main():
    launch=read(HERE/'latest_beauty_run.json')
    group=Path(launch['group'])
    status=read(group/'job_status.json')
    progress=read(group/'formal/progress.json')
    preflight=read(group/'preflight/stage_result.json')
    report=dict(observed_at=time.time(),launch=launch,status=status,
                supervisor_alive=Path(f'/proc/{status["supervisor_pid"]}').exists(),
                child_alive=Path(f'/proc/{status["pid"]}').exists() if status.get('pid') else False,
                preflight_passed=bool(preflight and preflight.get('completed')),
                preflight_modes=list(preflight['test_metrics']) if preflight else [],
                formal_model_verified=read(group/'formal/model_verified.json'),
                formal_progress=progress,formal_result=read(group/'formal/stage_result.json'),
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                             '--format=csv'],text=True),
                formal_log_tail=(group/'formal.log').read_text(errors='replace').splitlines()[-12:]
                if (group/'formal.log').exists() else [])
    (HERE/'last_status_snapshot.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
