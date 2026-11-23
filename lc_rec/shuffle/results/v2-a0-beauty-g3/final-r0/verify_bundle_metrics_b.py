"""Re-run the fixed independent full verifier against an extracted G3 delivery."""
from argparse import ArgumentParser
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    bundle = args.bundle.resolve(strict=True)
    output = args.output.absolute()
    script = Path(__file__).with_name('verify_full_metrics_b.py')
    assert not output.exists() and output.parent.is_dir()
    with TemporaryDirectory(prefix='g3-metrics-layout-') as temporary:
        layout = Path(temporary)
        links = {'cpu-normal-r0': bundle/'evaluation/normal',
                 'cpu-formal-paired-r0': bundle/'evaluation/paired'}
        for mode in ('shuffle2', 'shuffle3a', 'shuffle3b', 'shuffle4'):
            links[f'gpu-formal-{mode}-r0'] = bundle/'predictions'/mode
            links[f'cpu-formal-eval-{mode}-r0'] = bundle/'evaluation'/mode
        for name, source in links.items():
            assert source.is_dir(), source
            (layout/name).symlink_to(source, target_is_directory=True)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
        env.pop('PYTHONOPTIMIZE', None)
        subprocess.run([sys.executable, '-B', str(script),
                        '--root', str(layout), '--data', str(bundle/'data/Beauty'),
                        '--normal-test', str(bundle/'predictions/normal'),
                        '--archive', str(bundle/'frozen/normal-archived-primary'),
                        '--added-tokens', str(bundle/'frozen/adapter/added_tokens.json'),
                        '--output', str(output)], check=True, env=env)


if __name__ == '__main__':
    main()
