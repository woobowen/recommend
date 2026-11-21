"""Build the scoped launch-observation ZIP from fixed sources and saved evidence."""
import argparse
import difflib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import zipfile

from lc_rec.shuffle_v2.beauty_package import check_navigation, create

RUN_COMMIT = '1182dfd914c20bd3dc11786587eae6879fba6b9d'
OPERATOR_COMMIT = 'a2a684625ff61f79967ec4152de98526b52936f2'
ORIGINAL_BEAUTY = '6e1b211d9c37f358c43b1540b40b17f6703150d2'
RESULT = 'lc_rec/shuffle/results/v2-a0-beauty-letter/launch-observation-r0'
STATUS = 'SECOND_GROUP_FORMAL_RUNNING_30MIN_OBSERVATION_PASS'
V2 = 'lc_rec/shuffle_v2/'
RUN_FILES = {
    V2 + item for item in ('beauty.py','a0_beauty_letter.json','beauty_evaluate.py','beauty_runner.py',
                           'core.py','evaluate.py','generate.py','reference.py','replay.py','runtime.py',
                           'series.py','statistics.py')
} | {'lc_rec/shuffle/core.py', 'lc_rec/letter/scripts/formal_evaluate.py',
     'lc_rec/letter/review/final_audit_20261113/statistics_verify.py'}
OPERATOR_FILES = {V2 + item for item in ('beauty_control.py','control.py','collect.py','beauty_observe.py','beauty_package.py')}
OBS_FILES = {
    V2+item for item in ('beauty_formal_observe.py','beauty_formal_collect.py','beauty_formal_samples.py',
                         'beauty_formal_summary.py','beauty_formal_bundle.py')
}
FROZEN_FILES = {
    'lc_rec/a0_beauty/'+item for item in ('formal_evaluate.py','offline_metrics.py','uneven_sampler.py','shuffle_entry.py')
} | {'lc_rec/letter/review/final_audit_20261113/external/official_cce.py'}
REPO = Path(__file__).resolve().parents[2]
SUPPORT_SCRIPTS = Path('/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
OFFLINE_REVIEW = '''#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 1
root="$(cd -- "$(dirname -- "$0")" && pwd)"
destination="$1"
mkdir -- "$destination"
destination="$(cd -- "$destination" && pwd)"
cd -- "$root"
export PYTHONPATH="$root/source:$root/source/support"
python3 -B -m lc_rec.shuffle_v2.beauty_formal_samples evidence/samples/formal-first-six.json > "$destination/prefix.json"
python3 -B -m lc_rec.shuffle_v2.beauty_formal_summary --observations evidence/observations --plan evidence/launch/plan.json --assessment evidence/launch/preflight-assessment.json > "$destination/window.json"
python3 -B - "$destination" <<'VERIFY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for produced, saved in [('prefix.json','evidence/samples/formal-first-six-verified.json'),
                        ('window.json','evidence/observations/summary.json')]:
    assert json.loads((root/produced).read_text()) == json.loads(Path(saved).read_text()), produced
print('PASS: six exact prefix records and seven timed observations reproduced')
VERIFY
'''



def git(*args):
    return subprocess.check_output(['git',*args],cwd=REPO)


def add_sources(commit,directory,members):
    categories=((RUN_COMMIT,'source',RUN_FILES),(OPERATOR_COMMIT,'source',OPERATOR_FILES),
                (commit,'source',OBS_FILES),(ORIGINAL_BEAUTY,'archive/frozen-beauty',FROZEN_FILES))
    for version,area,files in categories:
        for file in sorted(files):
            name=area+'/'+file
            target=directory/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(git('show',f'{version}:{file}'))
            members[name]=str(target)
    vendor='lc_rec/letter/vendor/LETTER/LETTER-LC-Rec/'
    for file in ('data.py','collator.py','evaluate.py','prompt.py','utils.py'):
        name='archive/frozen-beauty/LETTER-LC-Rec/'+file
        target=directory/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(git('show',f'{ORIGINAL_BEAUTY}:{vendor}{file}'))
        members[name]=str(target)
    for file in ('gridctl.py','transport.py'):
        members['source/support/'+file]=str(SUPPORT_SCRIPTS/file)
    legacy='lc_rec/a1_qwen3_06b/scripts/runner.py'
    old=git('show',f'{RUN_COMMIT}:{legacy}').decode().splitlines(True)
    new=(directory/'source'/V2/'beauty_runner.py').read_text().splitlines(True)
    delta=directory/'runner-beauty.diff'
    delta.write_text(''.join(difflib.unified_diff(old,new,fromfile=legacy,tofile=V2+'beauty_runner.py')))
    members['source/runner-beauty.diff']=str(delta)


def build(args):
    commit=args.publication
    assert git('rev-parse','HEAD').decode().strip()==commit
    assert not args.output.exists()
    stage=REPO/RESULT
    assert stage.is_dir() and args.readback.is_file()
    summary=json.loads((stage/'observations/summary.json').read_text())
    assert summary['status']=='PASS_30MIN_FORMAL_BEAUTY_OBSERVATION'
    members={'README.md':str(args.readme),'publication/github-readback.json':str(args.readback)}
    for path in sorted(stage.rglob('*')):
        assert not path.is_symlink(),path
        if path.is_file():
            relative=path.relative_to(stage).as_posix()
            assert path.read_bytes()==git('show',f'{commit}:{RESULT}/{relative}')
            members['evidence/'+relative]=str(path)
    with TemporaryDirectory(prefix='beauty-formal-stage-') as work:
        add_sources(commit,Path(work),members)
        offline=Path(work)/'offline_review.sh'
        offline.write_text(OFFLINE_REVIEW)
        members['offline_review.sh']=str(offline)
        required={f'evidence/observations/t{minute:02}.json' for minute in range(0,31,5)}
        required |= {'evidence/AUTHORIZATION.md','evidence/launch/plan.json',
                     'evidence/samples/formal-first-six.json','evidence/launch/delivery-observation.json',
                     'evidence/reviews/window-nonauthor.json','evidence/reviews/launch-nonauthor.json',
                     'evidence/reviews/first-samples-nonauthor.json'}
        assert required <= members.keys(),sorted(required-members.keys())
        receipt=create({'status':STATUS,'files':members},args.output)
    with zipfile.ZipFile(args.output) as archive:
        check_navigation(archive)
    return receipt


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--publication',required=True)
    p.add_argument('--readme',type=Path,required=True)
    p.add_argument('--readback',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(build(args),indent=2))


if __name__=='__main__':
    main()
