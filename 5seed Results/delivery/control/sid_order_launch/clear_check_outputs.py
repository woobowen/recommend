"""Remove the explicitly named new-code checks after saving verification."""
from pathlib import Path
import shutil

root = Path('/data/jcyin/sid_order_reproduction')
names = ['new_code_check_20261015', 'new_attention_check_20261015',
         'new_attention_check_20261015_v2']
for name in names:
    for suffix in ('', '.log'):
        path = root / (name + suffix)
        assert not path.is_symlink() and path.resolve().parent == root
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()
        print(f'Cleared {path}')
