"""Durable lightweight evidence written after collective checkpoint completion.

This function consumes the already-hashed completion manifest; it never reads
model tensors or training/RNG state.  It must run before checkpoint rotation.
"""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile


def write_receipt(checkpoint, output, *, run_commit, config_sha256, data_identity,
                  epoch, step, eval_loss, best_checkpoint, best_metric, attempt,
                  required_files, expected_manifest):
    cp = Path(checkpoint)
    raw = (cp / 'complete.json').read_bytes()
    manifest = json.loads(raw)
    if manifest != expected_manifest:
        raise ValueError('checkpoint completion manifest changed')
    if (manifest.get('run_commit') != run_commit or
        manifest.get('config_sha256') != config_sha256 or
        manifest.get('global_step') != step or manifest.get('collective_ranks') != 2):
        raise ValueError('checkpoint completion identity mismatch')
    if not re.fullmatch(r'[0-9a-f]{40}', run_commit) or not re.fullmatch(r'[0-9a-f]{64}', config_sha256):
        raise ValueError('invalid run or config identity')
    if not isinstance(attempt, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,90}', attempt):
        raise ValueError('save attempt identity required')
    if not isinstance(data_identity, dict) or not data_identity:
        raise ValueError('data identity required')
    if not math.isfinite(float(epoch)) or not math.isfinite(float(eval_loss)):
        raise ValueError('invalid epoch or loss')
    if best_metric is not None and not math.isfinite(float(best_metric)):
        raise ValueError('invalid best metric')
    rows = manifest.get('files')
    if not isinstance(rows, list) or not rows:
        raise ValueError('checkpoint file manifest missing')
    seen = set()
    for row in rows:
        name = row.get('path')
        if not isinstance(name, str):
            raise ValueError('invalid manifest path')
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or str(p) != name or name in seen:
            raise ValueError('unsafe or duplicate manifest path')
        if not isinstance(row.get('bytes'), int) or row['bytes'] <= 0:
            raise ValueError('invalid manifest size')
        if not isinstance(row.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', row['sha256']):
            raise ValueError('invalid manifest hash')
        seen.add(name)
    if not set(required_files) <= seen:
        raise ValueError('incomplete checkpoint manifest')
    receipt = {'schema': 'beauty_checkpoint_receipt_v1', 'run_commit': run_commit,
               'config_sha256': config_sha256, 'data_identity': data_identity,
               'epoch': epoch, 'global_step': step, 'eval_loss': float(eval_loss),
               'best_checkpoint': best_checkpoint, 'best_metric': best_metric,
               'checkpoint_path': str(cp.resolve()), 'attempt_job_id': attempt,
               'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'manifest': manifest}
    payload = (json.dumps(receipt, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False) + '\n').encode()
    directory = Path(output) / 'checkpoint_receipts'
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / ('step-%d-%s.json' % (step, attempt))
    fd, tmp = tempfile.mkstemp(prefix='.receipt-', dir=directory)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(tmp, dest)  # atomic create-if-absent; never replace old evidence
        except FileExistsError:
            if dest.read_bytes() != payload:
                raise ValueError('conflicting receipt for same save identity')
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.unlink(tmp)
    return dest
