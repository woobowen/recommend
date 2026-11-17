"""Print the six completed experiments; use stored values without rerunning models."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def main():
    rows = json.loads((BASE / 'results/embedding_summary.json').read_text(encoding='utf-8'))
    print('| Dataset | Input | Delta12 | R+12 | Delta23 | Rmono | C3 |')
    print('|---|---|---:|---:|---:|---:|---:|')
    for r in rows:
        print(f"| {r['dataset']} | {r['variant']} | {r['delta_12']:+.6f} | "
              f"{r['positive_12']:.2%} | {r['delta_23']:+.6f} | "
              f"{r['path_monotonic']:.2%} | {r['coverage_3']:.2%} |")


if __name__ == '__main__':
    main()
