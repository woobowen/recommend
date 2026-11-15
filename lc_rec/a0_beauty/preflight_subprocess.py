"""Bounded subprocess supervision for the Beauty preflight runner."""
import subprocess
import time


def run(argv, deadline, stage, observe, poll_interval=5):
    child=subprocess.Popen(argv)
    try:
        while child.poll() is None:
            remaining=deadline-time.monotonic()
            if remaining<=0:
                raise TimeoutError('preflight wall budget exhausted')
            observe()
            try:child.wait(timeout=min(poll_interval,remaining))
            except subprocess.TimeoutExpired:pass
        stage['exit_code']=child.returncode
        if child.returncode:
            raise subprocess.CalledProcessError(child.returncode,argv)
    except BaseException as error:
        stage['interrupted']=repr(error)
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=8)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        stage['exit_code']=child.returncode
        raise
