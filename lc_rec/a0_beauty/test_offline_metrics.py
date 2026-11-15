"""Test inherited ranking and CCE fixtures; not a full prediction audit."""
from pathlib import Path
import hashlib
import json
import offline_metrics


def main():
    repo = Path(__file__).resolve().parents[2]
    path = repo / 'lc_rec/letter/review/final_audit_20261113/external/official_cce.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '173c257a8711a885e8f6d5abcdb0ad2510ad4fab01c1700631b5ec261f50cd2d'
    result = offline_metrics.selftest(offline_metrics.module(path))
    assert result['status'] == 'PASS' and result['numerical_assertions'] == 127
    print(json.dumps({'status': 'PASS', 'numerical_assertions': result['numerical_assertions'],
                      'cce_cases': len(result['cce_cases']),
                      'scope': 'Synthetic inherited fixtures; no Beauty predictions'}))


if __name__ == '__main__':
    main()
