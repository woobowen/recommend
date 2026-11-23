"""Compact read-only snapshot of both official LETTER downstream jobs."""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
HERE=ROOT/'codex/letter_official_downstream_20261116'


def read(path):return json.loads(path.read_text()) if path.is_file() else None


def main():
    manifest=read(HERE/'latest_queue.json');group=Path(manifest['group'])
    tasks=[]
    for task in manifest['tasks']:
        name=task['dataset'];status=read(group/f'{name}_status.json')
        check=read(group/name/'preflight/stage_result.json')
        formal=group/name/'formal'
        item=dict(dataset=name,status=status,
            supervisor_alive=Path(f'/proc/{status["supervisor_pid"]}').exists() if status.get('supervisor_pid') else False,
            child_alive=Path(f'/proc/{status["pid"]}').exists() if status.get('pid') else False,
            preflight_passed=bool(check and check.get('completed')),
            preflight_modes=list(check['test_metrics']) if check else [],
            model_verified=read(formal/'model_verified.json'),progress=read(formal/'progress.json'),
            result=read(formal/'stage_result.json'))
        stage=status.get('stage','preflight')
        log=group/f'{name}_{stage}.log'
        item['log_tail']=log.read_text(errors='replace').splitlines()[-12:] if log.exists() else []
        tasks.append(item)
    result=dict(observed_at=time.time(),group=str(group),tasks=tasks,
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                             '--format=csv'],text=True))
    (HERE/'last_status_snapshot.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
