"""CPU-only receipt tests; no model, optimizer, GPU or formal output."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from checkpoint_receipt import write_receipt


class ReceiptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.commit = 'a' * 40
        self.config = 'b' * 64

    def make(self, step, ranks=2):
        cp = self.root / ('checkpoint-%d' % step)
        cp.mkdir(exist_ok=True)
        manifest = {'run_commit': self.commit, 'config_sha256': self.config,
                    'global_step': step, 'collective_ranks': ranks,
                    'files': [{'path': 'file', 'bytes': 1, 'sha256': 'c' * 64}]}
        (cp / 'complete.json').write_text(json.dumps(manifest))
        return cp, manifest

    def save(self, cp, manifest, **kw):
        args = dict(run_commit=self.commit, config_sha256=self.config,
                    data_identity={'asset': {'sha256': 'd' * 64}}, epoch=1.0,
                    step=manifest['global_step'], eval_loss=0.5, best_checkpoint=str(cp),
                    best_metric=0.5, attempt='test-attempt', required_files={'file'},
                    expected_manifest=manifest)
        args.update(kw)
        return write_receipt(cp, self.root, **args)

    def test_persistence_and_conflict(self):
        cp, manifest = self.make(2053)
        path = self.save(cp, manifest)
        original = path.read_bytes()
        self.assertEqual(self.save(cp, manifest), path)
        with self.assertRaises(ValueError):
            self.save(cp, manifest, eval_loss=0.7)
        self.assertEqual(path.read_bytes(), original)

    def test_manifest_corruption_and_incomplete_rank(self):
        cp, manifest = self.make(2053)
        damaged = dict(manifest, global_step=3)
        (cp / 'complete.json').write_text(json.dumps(damaged))
        with self.assertRaises(ValueError):
            self.save(cp, manifest)
        cp, manifest = self.make(4107, ranks=1)
        with self.assertRaises(ValueError):
            self.save(cp, manifest)

    def test_write_failure_is_propagated(self):
        cp, manifest = self.make(2053)
        with patch('checkpoint_receipt.os.link', side_effect=OSError('injected')):
            with self.assertRaises(OSError):
                self.save(cp, manifest)
        self.assertFalse(list((self.root / 'checkpoint_receipts').glob('*.json')))

if __name__ == '__main__':
    unittest.main()
