import torch
import wandb
import os
from pathlib import Path
import signal
from types import FrameType
from openood.utils.config import setup_config
from openood.pipelines import get_pipeline

config = setup_config()
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = get_pipeline(config)
pipeline.run()
