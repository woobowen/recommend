"""Directed author checks; separate nonauthor review remains required."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from lc_rec.shuffle import core


class Tests(unittest.TestCase):
    def test_reference_and_rejection(self):
        for n in (3, 4):
            catalog = tuple(p for p in itertools.permutations(range(n)) if p != tuple(range(n)))
            self.assertEqual(len(catalog), {3: 5, 4: 23}[n])
            for row in (0, 12385, 12386, 24771):
                for item in (0, 19):
                    payload = ('["GRID-LETTER-SHUFFLE-V1",42,%d,%d,%d,0]' % (n, row, item)).encode()
                    value = int.from_bytes(hashlib.sha256(payload).digest(), 'big')
                    self.assertLess(value, (2**256//len(catalog))*len(catalog))
                    self.assertEqual(core.permutation(n, row, item), catalog[value % len(catalog)])
        with patch.object(core, '_digest256', side_effect=[b'\xff'*32, (7).to_bytes(32, 'big')]):
            self.assertEqual(core._draw(3, 0, 0, 42), (2, (1, 2, 0), 1))

    def test_realistic_boundaries_conservation_padding_and_replay(self):
        for count in (0, 1, 2, 20):
            groups = [[10, 11, 12, 13]]*count
            ids = [0, 1, 2]+sum((g+[3, 4] for g in groups), [])+[5, 0]
            mask = [0]+[1]*(len(ids)-2)+[0]
            for n in (0, 3, 4):
                new, ev = core.transform(ids, mask, groups, range(10, 14), n, 24771)
                replay = ids.copy()
                for g in ev['groups']:
                    for pos, v in zip(g['positions'], [g['original_ids'][j] for j in g['permutation']]):
                        replay[pos] = v
                    if n == 3: self.assertEqual(g['permutation'][3], 3)
                self.assertEqual(replay, new)
                self.assertCountEqual(ids, new)
                self.assertEqual(ids[0], new[0]);self.assertEqual(ids[-1], new[-1])
                self.assertEqual(new == ids, n == 0 or count == 0)
        with self.assertRaises(ValueError):
            core.transform([1,10,11,2,12,13], [1]*6, [[10,11,12,13]], range(10,14), 3, 0)
        with self.assertRaises(ValueError):
            core.transform([1,10,11,12,13,10], [1]*6, [[10,11,12,13]], range(10,14), 3, 0)

    def test_rank_order_and_python_hash_independence(self):
        values = {(r,j): core.permutation(4,r,j) for r in range(32) for j in range(20)}
        self.assertEqual(values, {(r,j): core.permutation(4,r,j) for r in reversed(range(32)) for j in reversed(range(20))})
        program = 'from lc_rec.shuffle.core import permutation; print([permutation(n,r,j) for n in (3,4) for r in (0,24771) for j in (0,19)])'
        outputs = [subprocess.check_output([sys.executable,'-c',program],
                   cwd=Path(__file__).resolve().parents[2], env=dict(os.environ, PYTHONHASHSEED=s)) for s in ['1','1234']]
        self.assertEqual(*outputs)


if __name__ == '__main__': unittest.main()
