"""Task-scoped Windows OpenSSH transport. No credentials are accepted or logged."""
from pathlib import Path, PurePosixPath
import datetime as dt
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]
SSH = '/mnt/c/Windows/System32/OpenSSH/ssh.exe'
SCP = '/mnt/c/Windows/System32/OpenSSH/scp.exe'
STAGE = Path('/mnt/c/Users/lenovo/AppData/Local/Temp/sid-extensions-p0')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def atomic(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    os.replace(temp, path)

class Transport:
    def __init__(self, target):
        self.target = target
        self.profile = json.loads((ROOT/'profiles.json').read_text())[target]
        self.host = self.profile['host_alias']
        self.options = ['-F','C:/Users/lenovo/.ssh/config','-o','BatchMode=yes',
            '-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',
            '-o','ClearAllForwardings=yes','-o','PermitLocalCommand=no',
            '-l',self.profile['user']]
        if self.profile.get('identity_file'):
            self.options += ['-i',self.profile['identity_file'],'-o','IdentitiesOnly=yes']
        if target == 'A800':
            # Preserve the exact authentication boundary of the two successful
            # precheck connections; each operation opens a fresh connection.
            self.options += ['-o','PreferredAuthentications=publickey',
                '-o','PasswordAuthentication=no','-o','KbdInteractiveAuthentication=no',
                '-o','IdentityAgent=none','-o','ControlMaster=no',
                '-o','ControlPath=none','-o','ControlPersist=no']
            if self.profile.get('jump_host'):
                self.options += ['-J', self.profile['jump_host']]

    def run(self, args, data=None, timeout=45):
        begin = dt.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        try:
            p = subprocess.run(args,input=data,capture_output=True,timeout=timeout)
            result = dict(started=begin,target=self.target,args=args,returncode=p.returncode,
                stdout=p.stdout.decode('utf-8','replace'),stderr=p.stderr.decode('utf-8','replace'))
        except subprocess.TimeoutExpired as e:
            result = dict(started=begin,target=self.target,args=args,returncode=None,
                error='transport timeout; remote state unknown; query same ID before retry',
                stdout=(e.stdout or b'').decode('utf-8','replace'),stderr=(e.stderr or b'').decode('utf-8','replace'))
        name = begin.replace(':','').replace('+','_')+'-'+uuid.uuid4().hex[:8]
        evidence=ROOT/'evidence'/'transport'/self.target
        if data is not None:
            evidence.mkdir(parents=True,exist_ok=True)
            (evidence/(name+'.stdin')).write_bytes(data)
            result['stdin_sha256']=hashlib.sha256(data).hexdigest()
        atomic(evidence/(name+'.json'),result)
        if result['returncode'] != 0: raise RuntimeError(json.dumps(result,ensure_ascii=False))
        return result['stdout']

    def ssh(self, command, data=None, timeout=45):
        return self.run([SSH,*self.options,'-T',self.host,command],data,timeout)

    def python(self, code, timeout=45):
        return self.ssh('python3 -I -S -B -',code.encode(),timeout)

    def identity(self):
        got=json.loads(self.python("import os,pwd,socket,json,sys; print(json.dumps(dict(hostname=socket.gethostname(),uid=os.getuid(),user=pwd.getpwuid(os.getuid()).pw_name,home=os.path.expanduser('~'),python=sys.executable,version=sys.version)))"))
        for key in ['hostname','uid','user']:
            if self.profile.get(key) is not None and got[key]!=self.profile[key]:
                raise RuntimeError('remote identity mismatch: '+key)
        return got

    def winpath(self,p):
        return subprocess.check_output(['wslpath','-w',str(p)],text=True).strip().replace('\\','/')

    def write_path(self, remote):
        root=self.profile.get('remote_support')
        p=PurePosixPath(remote)
        if not root or not p.is_absolute() or '..' in p.parts or not str(p).startswith(root+'/'):
            raise ValueError('write outside task-specific remote support root')
        return root

    def put(self, local, remote):
        root=self.write_path(remote)
        local=Path(local).resolve(); digest=sha(local); size=local.stat().st_size
        STAGE.mkdir(parents=True,exist_ok=True)
        stage=STAGE/(uuid.uuid4().hex+'-'+local.name); shutil.copyfile(local,stage)
        partial=remote+'.'+uuid.uuid4().hex+'.partial'
        self.python("from pathlib import Path\np=Path("+repr(remote)+");r=Path("+repr(root)+")\nassert not r.is_symlink()\nr.mkdir(parents=True,exist_ok=True)\nassert r.resolve() in p.resolve().parents\np.parent.mkdir(parents=True,exist_ok=True)\n")
        # SCP has its own -l meaning; use explicit User= for the account.
        scp_options=self.options.copy(); idx=scp_options.index('-l'); scp_options[idx:idx+2]=['-o','User='+self.profile['user']]
        self.run([SCP,*scp_options,self.winpath(stage),self.host+':'+shlex.quote(partial)],timeout=120)
        result=self.python("from pathlib import Path\nimport hashlib,os,json\np=Path("+repr(partial)+");q=Path("+repr(remote)+")\nh=hashlib.sha256(p.read_bytes()).hexdigest()\nassert p.stat().st_size=="+str(size)+" and h=="+repr(digest)+"\nif q.exists():\n assert q.is_file() and q.stat().st_size=="+str(size)+" and hashlib.sha256(q.read_bytes()).hexdigest()==h\n p.unlink()\nelse:\n os.link(p,q);p.unlink()\nprint(json.dumps(dict(path=str(q),bytes="+str(size)+",sha256=h,verified=True)))\n")
        stage.unlink(); return json.loads(result)

    def get_verified(self, remote, local, expected, timeout=300):
        # Fixed, hash-bound chunks resume at the completed-file boundary.
        local=Path(local).resolve(); local.parent.mkdir(parents=True,exist_ok=True)
        if local.exists() and local.stat().st_size==expected['bytes'] and sha(local)==expected['sha256']:
            return dict(expected,local=str(local),remote=remote,resumed=True,verified=True)
        STAGE.mkdir(parents=True,exist_ok=True)
        tag=hashlib.sha256((self.target+'\0'+remote+'\0'+expected['sha256']).encode()).hexdigest()
        stage=STAGE/(tag+'.download.partial')
        scp_options=self.options.copy();idx=scp_options.index('-l');scp_options[idx:idx+2]=['-o','User='+self.profile['user']]
        self.run([SCP,*scp_options,self.host+':'+shlex.quote(remote),self.winpath(stage)],timeout=timeout)
        if stage.stat().st_size!=expected['bytes'] or sha(stage)!=expected['sha256']:
            raise RuntimeError('download identity mismatch; retain partial for diagnosis')
        temporary=local.with_name(local.name+'.partial');shutil.copyfile(stage,temporary)
        if sha(temporary)!=expected['sha256']: raise RuntimeError('destination identity mismatch')
        os.replace(temporary,local);stage.unlink()
        result=dict(expected,local=str(local),remote=remote,resumed=False,verified=True)
        atomic(local.with_name(local.name+'.receipt.json'),result);return result
