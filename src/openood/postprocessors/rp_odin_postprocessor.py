from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .odin_postprocessor import ODINPostprocessor


class RPODINPostprocessor(ODINPostprocessor):
    def __init__(self, config):
        super(RPODINPostprocessor, self).__init__(config)
        self.targets = None  # will be updated in Evaluator

    def postprocess(self, net: nn.Module, data: Any):
        data.requires_grad = True
        output = net(data)
        softmax_output = torch.softmax(output, dim=-1)
        softmax_output = softmax_output.data.cpu()
        softmax_output = softmax_output.numpy()
        sim = -softmax_output * self.targets.cpu().numpy()
        sim = sim.sum(axis=1) / (
            np.linalg.norm(softmax_output, axis=-1) * np.linalg.norm(self.targets.cpu().numpy(), axis=-1)
        )
        sim = np.expand_dims(sim, axis=1)
        sim = sim + 1
        sim = torch.from_numpy(sim).cuda()

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        criterion = nn.CrossEntropyLoss()

        labels = output.detach().argmax(axis=1)

        # Using temperature scaling
        output = output / self.temperature

        loss = criterion(output, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(data.grad.detach(), 0)
        gradient = (gradient.float() - 0.5) * 2

        # Scaling values taken from original code
        gradient[:, 0] = (gradient[:, 0]) / self.input_std[0]
        gradient[:, 1] = (gradient[:, 1]) / self.input_std[1]
        gradient[:, 2] = (gradient[:, 2]) / self.input_std[2]

        # Adding small perturbations to images
        tempInputs = torch.add(data.detach(), gradient, alpha=-self.noise)
        output = net(tempInputs)
        output = output / self.temperature

        # Calculating the confidence after adding perturbations
        nnOutput = output.detach()
        nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
        nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)
        nnOutput = sim * nnOutput
        conf, pred = nnOutput.max(dim=1)

        return pred, conf
