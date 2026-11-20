import concurrent.futures
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from lc_rec.shuffle_v2 import core, reference


class Protocol(unittest.TestCase):
    def test_permutations_and_mapping(self):
        for mode, active in core.MODES.items():
            old = [40, 70, 57, 93]
            for index, permutation in enumerate(itertools.permutations(range(len(active)))):
                with patch.object(core, 'draw', return_value=(index, permutation, 0)):
                    new, evidence = core.transform([1, *old, 2], [1] * 6, [old], old, mode, 0)
                self.assertEqual(new, [1, *reference.remap(old, active, permutation), 2])
                self.assertEqual(evidence['input_changed'], index != 0)
                self.assertEqual(evidence['groups'][0]['nonidentity'], index != 0)
                for pos in set(range(4)) - set(active):
                    self.assertEqual(old[pos], new[pos + 1])

    def test_reference_and_rejection(self):
        for m, row, occurrence in itertools.product((2, 3, 4), range(22), range(21)):
            index, perm, rejected = core.draw(m, row, occurrence)
            self.assertEqual((index, list(perm), rejected), reference.relative(m, row, occurrence))
        watched = []
        def fake(payload):
            watched.append(payload)
            return bytes.fromhex('ff' * 32) if len(watched) == 1 else bytes(32)
        with patch.object(core, '_digest', side_effect=fake):
            self.assertEqual(core.draw(3, 9, 4), (0, (0, 1, 2), 1))
        self.assertEqual(watched, [b'["GRID-LETTER-SHUFFLE-V2",42,3,9,4,0]',
                                   b'["GRID-LETTER-SHUFFLE-V2",42,3,9,4,1]'])
        values = iter((bytes.fromhex('ff' * 32), bytes(32)))
        self.assertEqual(reference.relative(3, 9, 4, lambda _: SimpleNamespace(
            hexdigest=lambda: next(values).hex())), (0, [0, 1, 2], 1))

    def test_shared_relative_permutation_and_reordered_occurrences(self):
        old = [40, 70, 57, 93]
        repeated = [*old, 7, *old]
        def run(row, mode):
            return core.transform(repeated, [1] * len(repeated), [old, old], old, mode, row)[1]['groups']
        differs = False
        for row in range(24):
            a, b = run(row, 'shuffle3a'), run(row, 'shuffle3b')
            self.assertEqual([g['relative_permutation'] for g in a], [g['relative_permutation'] for g in b])
            differs |= a[0]['relative_permutation'] != a[1]['relative_permutation']
        self.assertTrue(differs)
        keys = list(itertools.product((2, 3, 4), range(8), range(9)))
        serial = {key: core.draw(*key) for key in keys}
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as workers:
            parallel = dict(zip(reversed(keys), workers.map(core.draw, *zip(*reversed(keys))), strict=True))
        self.assertEqual(serial, parallel)

    def test_boundaries(self):
        old = [40, 40, 57, 93]
        ids, mask = [0, 0, 1, *old, 2, 0], [0, 0, 1, 1, 1, 1, 1, 1, 0]
        changed, _ = core.transform(ids, mask, [old], old, 'shuffle4', 7)
        self.assertEqual(changed[:3], ids[:3])
        self.assertEqual(changed[-2:], ids[-2:])
        self.assertEqual(sorted(changed[3:7]), sorted(old))
        self.assertEqual(core.transform([0, 1], [0, 1], [], old, 'shuffle4', 7)[0], [0, 1])
        for bad_ids, bad_mask in ((ids[:-3], mask[:-3]), ([*ids, 40], [*mask, 0]), (ids, [0] * len(ids))):
            with self.assertRaises(ValueError):
                core.transform(bad_ids, bad_mask, [old], old, 'shuffle4', 7)
        with patch.object(core, 'draw', return_value=(6, (1, 0, 2, 3), 0)):
            _, ev = core.transform(ids, mask, [old], old, 'shuffle4', 7)
            self.assertTrue(ev['groups'][0]['nonidentity'])
            self.assertFalse(ev['input_changed'])


if __name__ == '__main__':
    unittest.main()
