from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gradnorm_postprocessor import GradNormPostprocessor


class RPGNPostprocessor(GradNormPostprocessor):
    def __init__(self, config):
        super(RPGNPostprocessor, self).__init__(config)
        self.targets = None  # will be updated in Evaluator

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # this follows the official implementation of class prior
        # https://github.com/tmlr-group/class_prior/blob/main/funcs.py#L182
        outputs, features = net.forward(data, return_feature=True)
        _, pred = torch.max(outputs, dim=1)
        U = torch.norm(features, p=1, dim=1)
        out_softmax = torch.nn.functional.softmax(outputs, dim=1)
        V = torch.norm((self.targets - out_softmax), p=1, dim=1)
        conf = U * V / 2048 / self.num_classes

        return pred, conf
