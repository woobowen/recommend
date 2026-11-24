"""Restore and hash the original formal NPY byte-for-byte from its uint16 representation."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def restore(folder, output):
    proof = json.loads((folder/'bootstrap-representation.json').read_bytes())
    summary = json.loads((folder/'summary.json').read_bytes())['bootstrap']
    packed_path = folder/'bootstrap_indices.uint16.npy'
    header = (folder/'bootstrap_indices.npy.header').read_bytes()
    assert len(header) == proof['header_bytes'] and sha256(header).hexdigest() == proof['header_sha256']
    assert packed_path.stat().st_size == proof['packed_bytes'] and digest(packed_path) == proof['packed_sha256']
    packed = np.load(packed_path, allow_pickle=False, mmap_mode='r')
    assert packed.shape == tuple(proof['shape']) == (5000,24772) and packed.flags.c_contiguous
    assert packed.dtype == np.dtype(proof['packed_dtype']) == np.dtype('<u2')
    assert proof['original_dtype'] == '<i8'
    raw_digest = sha256()
    full_digest = sha256(header)
    maximum = 0
    with output.open('xb') as out:
        out.write(header)
        for start in range(0, len(packed), 32):
            rows = packed[start:start+32]
            maximum = max(maximum,int(rows.max()))
            assert maximum < 24772
            wide = rows.astype('<i8').tobytes(order='C')
            raw_digest.update(wide)
            full_digest.update(wide)
            out.write(wide)
    assert output.stat().st_size == proof['original_bytes']
    assert raw_digest.hexdigest() == proof['original_index_data_sha256'] == summary['draw_indices_sha256_little_endian']
    assert full_digest.hexdigest() == proof['original_sha256'] == summary['draw_indices_file_sha256']
    restored = np.load(output, allow_pickle=False, mmap_mode='r')
    assert restored.dtype == np.dtype('<i8') and restored.shape == packed.shape
    assert restored.offset == len(header) and restored.flags.c_contiguous
    for start in range(0,len(packed),32):
        assert np.array_equal(restored[start:start+32],packed[start:start+32])
    assert digest(output) == full_digest.hexdigest()
    return dict(status='PASS_RESTORE_ORIGINAL_BOOTSTRAP_NPY', original_bytes=proof['original_bytes'],
                packed_bytes=proof['packed_bytes'], original_sha256=full_digest.hexdigest(),
                data_sha256=raw_digest.hexdigest(), dtype=restored.dtype.str, shape=list(restored.shape),
                values=restored.size, max_index=maximum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('paired', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(restore(args.paired,args.output),indent=2))
