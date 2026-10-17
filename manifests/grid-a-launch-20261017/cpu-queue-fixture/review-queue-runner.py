"""At-most-once Linux runner with verified, bounded foreground-group cleanup.

Schema/root/accepted-ID semantics are unchanged. No reboot or daemon recovery.
Only individually identity-checked task members receive signals, via pidfds.
"""
import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import time
import uuid

DEPLOYMENT_PATH = Path(str(Path(__file__).resolve()) + '.deployment.json')
DEPLOYMENT = json.loads(DEPLOYMENT_PATH.read_text())
JOB_ROOT = Path(DEPLOYMENT['job_root'])

TERM_GRACE_SECONDS = 2.0
KILL_GRACE_SECONDS = 3.0

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()

def digest(x):
    return hashlib.sha256(canonical(x)).hexdigest()

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def atomic(p, x):
    p = Path(p)
    tmp = p.with_name(p.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('wb') as f:
        f.write(canonical(x) + b'\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)
    fd = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

@contextlib.contextmanager
def lock(p, nonblock=False):
    with Path(p).open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblock else 0))
        yield

def identity(pid):
    try:
        p = Path('/proc') / str(pid)
        fields = (p/'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z': return None
        command = (p/'cmdline').read_bytes()
        if not command: return None
        return {'pid':pid, 'start_ticks':fields[19],
                'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'uid':p.stat().st_uid,
                'argv':command.decode('utf-8','replace').rstrip('\0').split('\0')}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None

def alive(saved):
    return bool(saved and identity(saved['pid']) == saved)

def proc_record(pid):
    """Minimal Linux identity, including zombies; never exports command/env."""
    p = Path('/proc') / str(pid)
    fields = (p/'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid':int(pid), 'start_ticks':fields[19], 'uid':p.stat().st_uid,
            'pgid':int(fields[2]), 'session_id':int(fields[3]), 'state':fields[0]}

def member_identity(record, boot_id):
    return {k:record[k] for k in ['pid','start_ticks','uid','pgid','session_id']} | {'boot_id':boot_id}

def same_member(record, saved, boot_id):
    return member_identity(record, boot_id) == {k:saved[k] for k in
            ['pid','start_ticks','uid','pgid','session_id','boot_id']}

def stable_member(record, saved, boot_id):
    # A proven process can change PGID/session or exec with different markers;
    # identity still belongs to the task, but such escapes are never signaled.
    return boot_id==saved['boot_id'] and all(record[k]==saved[k] for k in
                                           ['pid','start_ticks','uid'])

def workgroup(process, request, job):
    # Capture before launched_identity/poll can reap a very short-lived leader.
    rec = proc_record(process.pid)
    assert rec['pgid'] == rec['session_id'] == process.pid, 'worker did not enter its own session/group'
    assert rec['uid'] == os.getuid(), 'worker account changed before ownership capture'
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    leader = member_identity(rec, boot)
    return {'ownership_schema':1, 'source':'captured_at_launch', 'job_id':request['job_id'],
            'job_dir':str(job), 'boot_id':boot, 'uid':rec['uid'], 'pgid':rec['pgid'],
            'session_id':rec['session_id'], 'leader':leader, 'known_members':[leader]}

def legacy_workgroup(status, request, job):
    """Read-only interpretation of old runner's start_new_session contract."""
    saved = status.get('worker')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    group = {'ownership_schema':1, 'source':'legacy_start_new_session_inference' if saved else
             'legacy_marker_scan_no_recorded_group', 'job_id':request['job_id'], 'job_dir':str(job),
             'boot_id':saved['boot_id'] if saved else boot, 'uid':saved['uid'] if saved else os.getuid(),
             'pgid':saved['pid'] if saved else None, 'session_id':saved['pid'] if saved else None,
             'known_members':[]}
    if saved:
        member = {k:saved[k] for k in ['pid','start_ticks','uid','boot_id']}
        member.update(pgid=saved['pid'], session_id=saved['pid'])
        group['leader'] = member
        # Legacy identities contain no marker proof. Require markers before any
        # member can become owned; query/export never signal in any case.
    return group

def observe_group(group):
    """Observe original group and marked escapes even after leader exit.

    Unknown candidates and marked session escapes block completion/export.
    Zombies cannot continue work and are reported separately. Previously proven
    identities remain owned across exec/environment changes within the group.
    """
    now_boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    result = {'observed_at':utc(), 'source':group['source'], 'boot_changed':now_boot!=group['boot_id'],
              'owned_live':[], 'unconfirmed_live':[], 'unsupported_live':[], 'zombies':[],
              'scan_errors':[], 'known_members':list(group.get('known_members',[]))}
    if result['boot_changed']:
        result.update(scan_complete=True, verified_empty=True)
        return result
    known = {str(v['pid']):v for v in group.get('known_members',[])}
    markers = {('GRID_JOB_ID='+group['job_id']).encode(), ('GRID_JOB_DIR='+group['job_dir']).encode()}
    minimum = int(group.get('leader',{}).get('start_ticks',0))
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal(): continue
        owner = None
        try:
            owner = entry.stat().st_uid
            rec = proc_record(int(entry.name))
        except (FileNotFoundError,ProcessLookupError):
            continue
        except (PermissionError,OSError,ValueError,IndexError) as error:
            # Foreign-user unreadability does not imply uncertainty about this
            # account. An unreadable own-user record cannot be called clear.
            if owner is None or owner == group['uid']:
                result['scan_errors'].append({'pid':int(entry.name),'error':type(error).__name__})
            continue
        candidate = rec['pgid']==group['pgid'] and rec['session_id']==group['session_id']
        prior = known.get(entry.name)
        exact_known = bool(prior and stable_member(rec,prior,now_boot))
        if rec['uid'] != group['uid'] and not candidate: continue
        if rec['state'] == 'Z':
            if candidate or exact_known: result['zombies'].append(member_identity(rec,now_boot))
            continue
        try:
            env = set((entry/'environ').read_bytes().split(b'\0'))
            marked = markers <= env
        except (FileNotFoundError,ProcessLookupError):
            continue
        except (PermissionError,OSError) as error:
            marked = False
            if candidate or exact_known:
                result['unconfirmed_live'].append(dict(member_identity(rec,now_boot),
                                                      reason='ENVIRONMENT_UNREADABLE',error=type(error).__name__))
            continue
        item = member_identity(rec,now_boot)
        if candidate and rec['uid']==group['uid'] and int(rec['start_ticks'])>=minimum and (marked or exact_known):
            result['owned_live'].append(dict(item,proof='TASK_MARKERS' if marked else 'PREVIOUSLY_PROVEN_IDENTITY'))
            known[entry.name] = item
        elif candidate:
            result['unconfirmed_live'].append(dict(item,reason='GROUP_MEMBER_WITHOUT_OWNERSHIP_PROOF'))
        elif marked or exact_known:
            # We deliberately do not follow daemon/session escapes with signals.
            result['unsupported_live'].append(dict(item,reason='TASK_LEFT_SUPPORTED_GROUP'))
    result['known_members'] = sorted(known.values(),key=lambda v:(v['pid'],v['start_ticks']))
    result['scan_complete'] = not result['scan_errors']
    result['verified_empty'] = not any(result[k] for k in
                                      ['owned_live','unconfirmed_live','unsupported_live','scan_errors'])
    return result

def remember(group, observation):
    group['known_members'] = observation['known_members']

def public_observation(observation):
    return {k:v for k,v in observation.items() if k!='known_members'}

def signal_member(group, saved, signum):
    """No killpg/PID reuse window: bind pidfd, then recheck identity + proof."""
    if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
        raise RuntimeError('safe signaling requires Linux pidfd_open/pidfd_send_signal')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    try:
        fd = os.pidfd_open(saved['pid'],0)
    except ProcessLookupError:
        return 'ALREADY_EXITED'
    try:
        try: rec = proc_record(saved['pid'])
        except (FileNotFoundError,ProcessLookupError): return 'ALREADY_EXITED'
        if rec['state']=='Z': return 'ALREADY_ZOMBIE'
        if not same_member(rec,saved,boot):
            raise RuntimeError('refuse signal: member identity changed')
        if rec['uid']!=group['uid'] or rec['pgid']!=group['pgid'] or rec['session_id']!=group['session_id']:
            raise RuntimeError('refuse signal: member left supported ownership scope')
        known = any(same_member(rec,v,boot) for v in group.get('known_members',[]))
        if not known: raise RuntimeError('refuse signal: no prior marker proof')
        try: signal.pidfd_send_signal(fd,signum,None,0)
        except ProcessLookupError: return 'ALREADY_EXITED'
        return 'SENT'
    finally:
        os.close(fd)

def launched_identity(process, argv):
    # /proc/cmdline may briefly be empty while exec starts the new interpreter.
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        observed = identity(process.pid)
        if observed and observed['argv'] == argv:
            return observed
        if process.poll() is not None:
            return None
        time.sleep(0.01)
    raise RuntimeError('started process identity not confirmed; do not automatically restart')

def job_path(value):
    p=Path(value);root=JOB_ROOT
    if '..' in p.parts or p.parent!=root or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,90}',p.name):
        raise ValueError('job path outside setup root')
    if p.resolve().parent!=root.resolve():raise ValueError('resolved job path outside setup root')
    return p

def query(job):
    job = job_path(job)
    request = read(job/'request.json')
    verify_namespace(request)
    s = read(job/'status.json') if (job/'status.json').exists() else {
        'job_id':request['job_id'], 'status':'UNCONFIRMED', 'exit_code':None,
        'result_path':str(job/'output'), 'observation':'STATUS_MISSING_NO_AUTOMATIC_RESTART'}
    s['request_sha256'] = digest(request)
    s['supervisor_alive'] = alive(s.get('supervisor'))
    s['worker_alive'] = alive(s.get('worker'))
    group = s.get('worker_group') or legacy_workgroup(s,request,job)
    observation = observe_group(group)
    s['workgroup_observation'] = public_observation(observation)
    s['workgroup_live'] = bool(observation['owned_live'] or observation['unconfirmed_live'] or observation['unsupported_live'])
    s['cleanup_verified'] = observation['verified_empty']
    s['cleanup_evidence_level'] = group['source']
    if s.get('worker_group') and (s.get('cleanup') or {}).get('verified') is not True:
        s['cleanup_verified'] = False
    if s['status'] in ['ACCEPTED','RUNNING'] and not s['supervisor_alive']:
        s['persisted_status'] = s['status']
        s['status'] = 'CONTROL_LOST'
        s['observation'] = 'SUPERVISOR_MISSING_WORKERS_LIVE' if s['workgroup_live'] or s['worker_alive'] else 'UNCONFIRMED_NO_AUTOMATIC_RESTART'
        s['error'] = 'Recorded supervisor identity is absent; no automatic retry and no terminal success inferred'
        with lock(job/'.observation.lock'):
            atomic(job/'control-loss-observation.json', dict(observed_at=utc(), **{k:s.get(k) for k in ['job_id','request_sha256','persisted_status','status','observation','error','worker_alive','workgroup_live','supervisor_alive','supervisor']}))
    return s

def verify_namespace(r):
    import socket, pwd
    assert socket.gethostname() == r['host'] == DEPLOYMENT['hostname'], 'host identity mismatch'
    assert os.getuid() == r['uid'] == DEPLOYMENT['uid'], 'account UID mismatch'
    assert pwd.getpwuid(os.getuid()).pw_name == r['user'] == DEPLOYMENT['user'], 'account name mismatch'
    assert r['namespace'] == DEPLOYMENT['namespace'], 'task namespace mismatch'
    assert r['deployment_sha256'] == sha(DEPLOYMENT_PATH), 'deployment identity mismatch'
    assert isinstance(r['capture_version'],str) and r['capture_version'], 'input capture identity missing'

def verify(r):
    assert r['schema'] == 1
    verify_namespace(r)
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,90}', r['job_id'])
    assert isinstance(r['argv'], list) and r['argv'] and all(isinstance(v,str) for v in r['argv'])
    assert Path(r['argv'][0]).is_absolute() and os.access(r['argv'][0], os.X_OK)
    assert Path(r['cwd']).is_dir() and Path(r['cwd']).is_absolute()
    queue_only = (r['job_id'] == 'ga-20261017-queue-v1' and
        r.get('env', {}).get('CUDA_VISIBLE_DEVICES') == '' and
        r['argv'][1:5] == ['-B', '-m', 'experiments.launch_20261017.queue', 'supervise'])
    assert 0 < r['timeout_seconds'] <= (172800 if queue_only else 86400)
    assert all(isinstance(k,str) and isinstance(v,str) for k,v in r.get('env',{}).items())
    assert 'GRID_JOB_ID' not in r.get('env', {}) and 'GRID_JOB_DIR' not in r.get('env', {})
    for item in r['inputs']:
        p = Path(item['path'])
        assert p.is_absolute() and p.is_file() and sha(p) == item['sha256'], 'input changed: ' + str(p)

def submit(root, r):
    root = Path(root)
    assert root == JOB_ROOT
    verify_namespace(r)
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,90}', r['job_id'])
    root.mkdir(parents=True, exist_ok=True)
    job = root / r['job_id']
    assert job.resolve().parent == root.resolve(), 'job directory outside root'
    job.mkdir(exist_ok=True)
    with lock(job/'.submit.lock'):
        if (job/'request.json').exists():
            if digest(read(job/'request.json')) != digest(r):
                raise ValueError('REQUEST_ID_CONFLICT: same job_id, different immutable request')
        else:
            verify(r); atomic(job/'request.json',r)
        if (job/'accepted.json').exists():
            return dict(query(job), submission='EXISTING_NO_RESTART')
        verify(r)
        (job/'output').mkdir(exist_ok=True)
        atomic(job/'accepted.json',{'accepted_at':utc(),'request_sha256':digest(r),
                                   'runner_path':str(Path(__file__).resolve()),'runner_sha256':sha(__file__)})
        with lock(job/'.state.lock'):
            atomic(job/'status.json',{'job_id':r['job_id'],'status':'ACCEPTED','received_at':utc(),
                                      'exit_code':None,'result_path':str(job/'output')})
            with (job/'supervisor.log').open('ab') as log:
                supervisor_argv = [sys.executable,'-I','-S','-B',str(Path(__file__).resolve()),'supervise',str(job)]
                p = subprocess.Popen(supervisor_argv,
                                     stdin=subprocess.DEVNULL,stdout=log,stderr=log,cwd=job,
                                     start_new_session=True,close_fds=True)
            s = read(job/'status.json'); s['supervisor']=launched_identity(p,supervisor_argv); atomic(job/'status.json',s)
        return dict(query(job), submission='ACCEPTED')

def artifacts(job):
    return [{'path':str(p.relative_to(job)), 'bytes':p.stat().st_size, 'sha256':sha(p)}
            for p in sorted((job/'output').rglob('*')) if p.is_file() and not p.is_symlink()]

def stop_worker(p, group, job, reason):
    """Bounded TERM/KILL rounds; leader exit never terminates the group check."""
    report = {'reason':reason,'started_at':utc(),'events':[],'verified':False}
    def scan():
        p.poll()  # Reap only our direct child; never treats its exit as group exit.
        observed = observe_group(group);remember(group,observed)
        report['last_observation'] = public_observation(observed)
        return observed
    report['initial_observation'] = public_observation(scan())
    for signum,grace in [(signal.SIGTERM,TERM_GRACE_SECONDS),(signal.SIGKILL,KILL_GRACE_SECONDS)]:
        deadline = time.monotonic()+grace
        sent = set()
        while True:
            observed = scan()
            if observed['verified_empty']:
                report.update(verified=True,ended_at=utc())
                atomic(job/'cleanup.json',report)
                return report
            for member in observed['owned_live']:
                key = (member['pid'],member['start_ticks'])
                if key in sent: continue
                try: outcome = signal_member(group,member,signum)
                except Exception as error: outcome = 'REFUSED: '+repr(error)
                report['events'].append({'at':utc(),'member':member,'signal':int(signum),'outcome':outcome})
                sent.add(key)
            atomic(job/'cleanup.json',report)
            if time.monotonic()>=deadline: break
            time.sleep(0.05)
    report.update(verified=scan()['verified_empty'],ended_at=utc())
    atomic(job/'cleanup.json',report)
    return report

def supervise(job):
    job=job_path(job); p=None; saved=None; group=None; cleanup=None; interrupted=False
    def on_signal(signum,frame):
        nonlocal interrupted
        interrupted=True
    signal.signal(signal.SIGTERM,on_signal); signal.signal(signal.SIGINT,on_signal)
    with lock(job/'.run.lock',nonblock=True):
        try:
            r=read(job/'request.json'); verify(r)
            assert sha(__file__)==read(job/'accepted.json')['runner_sha256'], 'runner changed'
            # A durable claim survives supervisor death and prevents any second execution.
            with (job/'attempt.json').open('x') as f:
                json.dump({'claimed_at':utc(),'request_sha256':digest(r)},f); f.flush();os.fsync(f.fileno())
            os.environ['GRID_JOB_ID']=r['job_id']
            env=os.environ.copy();env.update(r.get('env',{}));env['GRID_JOB_DIR']=str(job)
            with lock(job/'.state.lock'):
                s=read(job/'status.json');s.update(supervisor=identity(os.getpid()),started_at=utc(),status='RUNNING')
                atomic(job/'status.json',s)
            with (job/'work.log').open('ab') as log:
                p=subprocess.Popen(r['argv'],cwd=r['cwd'],env=env,stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            group=workgroup(p,r,job)
            saved=launched_identity(p,r['argv'])
            with lock(job/'.state.lock'):
                s=read(job/'status.json');s.update(worker=saved,worker_group=group);atomic(job/'status.json',s)
            deadline=time.monotonic()+r['timeout_seconds'];timed_out=False
            while p.poll() is None:
                observed=observe_group(group)
                old_members=group['known_members'];remember(group,observed)
                if old_members!=group['known_members']:
                    with lock(job/'.state.lock'):
                        s=read(job/'status.json');s['worker_group']=group;atomic(job/'status.json',s)
                if interrupted or time.monotonic()>=deadline:
                    timed_out=not interrupted
                    cleanup=stop_worker(p,group,job,'INTERRUPTED' if interrupted else 'TIMED_OUT');break
                try:p.wait(timeout=min(0.25,max(0.01,deadline-time.monotonic())))
                except subprocess.TimeoutExpired:pass
            orphaned=False
            if cleanup is None:
                observed=observe_group(group);remember(group,observed)
                orphaned=not observed['verified_empty']
                cleanup=stop_worker(p,group,job,'LEADER_EXITED_WITH_MEMBERS' if orphaned else 'NORMAL_EXIT_CHECK')
            rc=p.poll()
            if rc is None:
                cleanup['verified']=False
                cleanup['leader_exit_unconfirmed']=True
                atomic(job/'cleanup.json',cleanup)
            outcome='CLEANUP_UNCONFIRMED' if not cleanup['verified'] else 'TIMED_OUT' if timed_out else 'INTERRUPTED' if interrupted else 'WORKERS_TERMINATED' if orphaned else 'SUCCEEDED' if rc==0 else 'FAILED'
            with lock(job/'.state.lock'):
                s=read(job/'status.json');s.update(exit_code=rc,ended_at=utc(),
                    status=outcome,worker_group=group,cleanup=cleanup,
                    worker_may_be_live=not cleanup['verified'])
                if cleanup['verified']:s['artifacts']=artifacts(job)
                atomic(job/'status.json',s)
        except BaseException as e:
            cleanup_error=None
            if p is not None and group is not None:
                try:cleanup=stop_worker(p,group,job,'SUPERVISOR_ERROR')
                except BaseException as error:cleanup_error=repr(error)
            worker_live=bool(p and (p.poll() is None or not cleanup or not cleanup['verified']))
            with lock(job/'.state.lock'):
                s=read(job/'status.json');s.update(status='SUPERVISOR_ERROR',ended_at=utc(),
                    error=repr(e),exit_code=p.poll() if p else None,worker_may_be_live=worker_live,
                    worker_group=group,cleanup=cleanup,cleanup_error=cleanup_error)
                atomic(job/'status.json',s)
            raise

def export(job):
    job=job_path(job);s=query(job)
    if s['status'] in ['ACCEPTED','RUNNING','UNCONFIRMED','CONTROL_LOST'] or s['worker_alive'] or s['workgroup_live'] or not s['cleanup_verified'] or s.get('worker_may_be_live'):
        raise RuntimeError('export only terminal jobs with verified empty task workgroup; query/log meanwhile')
    destination=job/'evidence.tar.gz'
    with lock(job/'.export.lock'):
        tmp=job/('evidence.'+uuid.uuid4().hex+'.partial')
        files=[p for p in sorted(job.rglob('*')) if p.is_file() and not p.is_symlink()
               and not p.name.endswith(('.partial','.tmp','.lock')) and p!=destination and p!=job/'export-manifest.json']
        atomic(job/'export-manifest.json',{'files':[{'path':str(p.relative_to(job)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]})
        with tarfile.open(tmp,'w:gz') as t:
            for p in files+[job/'export-manifest.json']:
                if p==job/'export-manifest.json' and p in files:continue
                t.add(p,arcname=str(p.relative_to(job)),recursive=False)
        os.replace(tmp,destination)
    return {'path':str(destination),'bytes':destination.stat().st_size,'sha256':sha(destination)}

if __name__=='__main__':
    os.umask(0o077)
    a=argparse.ArgumentParser();s=a.add_subparsers(dest='mode',required=True)
    p=s.add_parser('submit');p.add_argument('root');p.add_argument('request')
    for n in ['supervise','query','export']:
        p=s.add_parser(n);p.add_argument('job')
    n=a.parse_args()
    try:
        if n.mode=='submit':result=submit(n.root,read(n.request))
        elif n.mode=='supervise':supervise(n.job);result=None
        elif n.mode=='query':result=query(n.job)
        else:result=export(n.job)
        if result is not None:print(json.dumps(result,indent=2))
    except Exception as e:
        print(json.dumps({'error':repr(e)},indent=2),file=sys.stderr);sys.exit(2)
