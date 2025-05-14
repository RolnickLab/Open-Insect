import csv
import os
from typing import Dict, List
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.postprocessors import BasePostprocessor
from openood.postprocessors import OSRBranchPostprocessor
from openood.utils import Config
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_evaluator import BaseEvaluator
from .metrics import compute_all_metrics
import torch

from .osr_evaluator import OSREvaluator


class OSRBranchEvaluator(OSREvaluator):
    def __init__(self, config: Config):
        """OSR Branch Evaluator.

        Args:
            config (Config): Config file from
        """
        super(OSRBranchEvaluator, self).__init__(config)

    def eval_ood(
        self,
        net: nn.Module,
        id_data_loader: DataLoader,
        ood_data_loaders: Dict[str, Dict[str, DataLoader]],
        postprocessor: BasePostprocessor,
    ):
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()

        # load training in-distribution data
        assert "test" in id_data_loader, "id_data_loaders should have the key: test!"
        dataset_name = self.config.dataset.name
        print(f"Performing inference on {dataset_name} dataset...", flush=True)
        id_pred, id_conf, id_gt = postprocessor.inference(net, id_data_loader["test"])
        if self.config.recorder.save_scores:
            self._save_scores(id_pred, id_conf, id_gt, dataset_name)

        # load nearood data and compute ood metrics
        self._eval_ood(net, [id_pred, id_conf, id_gt], ood_data_loaders, postprocessor, ood_split="osr")
