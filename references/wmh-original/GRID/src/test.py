
from typing import Any, Dict, Optional, Tuple

import hydra
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator="./GRID", pythonpath=True)
from src.utils import RankedLogger, extras
from src.utils.custom_hydra_resolvers import *
from src.utils.launcher_utils import pipeline_launcher
from src.utils.restart_job import LocalJobLauncher