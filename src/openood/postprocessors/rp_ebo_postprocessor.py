from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ebo_postprocessor import EBOPostprocessor


class RPEBOPostprocessor(EBOPostprocessor):
    def __init__(self, config):
        super(RPEBOPostprocessor, self).__init__(config)
        self.targets = None  # will be updated in Evaluator

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        conf = self.temperature * torch.logsumexp(output / self.temperature, dim=1)
        sim = -score * self.targets
        sim = sim.sum(1) / (torch.norm(score, dim=1) * torch.norm(self.targets, dim=1))
        sim = sim + 1
        conf = conf * sim
        return pred, conf
