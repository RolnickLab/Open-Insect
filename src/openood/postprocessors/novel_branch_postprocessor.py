from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor


class NovelBranchPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(NovelBranchPostprocessor, self).__init__(config)
        self.config = config

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output, conf = net(data, return_pred_extended=True)
        # can either use logit or softmax
        # conf = torch.softmax(conf, dim=1) # use softmax
        conf = -conf[:, -1:].squeeze()
        _, pred = torch.max(output, dim=1)
        return pred, conf
