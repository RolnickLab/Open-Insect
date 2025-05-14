from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor


class RPMSPPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(RPMSPPostprocessor, self).__init__(config)
        self.targets = None  # will be updated in Evaluator

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        sim = score - self.targets
        conf, pred = torch.max(sim, dim=-1)
        return pred, conf
