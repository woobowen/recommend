"""Compatibility entry: current alignment means the historical GRID baseline."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().with_name('launch_grid_check.py')),run_name='__main__')
