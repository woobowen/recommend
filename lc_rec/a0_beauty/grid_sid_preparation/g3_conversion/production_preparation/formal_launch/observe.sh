#!/usr/bin/env bash
# The PID and snapshot mode are the only remote arguments. Source is streamed
# through a fresh SSH connection. This starts no computation.
set -euo pipefail
cd -- "$(dirname -- "$0")"
pid="${1:?controller PID}"
mode="${2:-sample}"
if [[ ! "$pid" =~ ^[1-9][0-9]*$ || ( "$mode" != sample && "$mode" != full ) ]]; then exit 2; fi
exec /mnt/c/Windows/System32/OpenSSH/ssh.exe -T -o BatchMode=yes -o ConnectTimeout=12 new-server   "/data/bwwu/envs/letter-20261110/bin/python -B - $pid $mode" < observe_remote.py
