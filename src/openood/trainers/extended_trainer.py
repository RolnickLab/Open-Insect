import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config

from .lr_scheduler import cosine_annealing
from timm.scheduler import CosineLRScheduler


class ExtendedTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader, config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config
        self.optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=config.optimizer["lr"],
            weight_decay=config.optimizer["weight_decay"],
        )

        steps_per_epoch = int(config.dataset.train.len / config.dataset.train.batch_size) + 1
        total_steps = int(config.optimizer.num_epochs * steps_per_epoch)
        warmup_steps = int(config.optimizer.warmup_epochs * steps_per_epoch)

        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=(total_steps - warmup_steps),
            warmup_t=warmup_steps,
            warmup_prefix=True,
            cycle_limit=1,
            t_in_epochs=False,
        )

    def train_epoch(self, epoch_idx):
        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)
        total_steps = (int(self.config.dataset.train.len / self.config.dataset.train.batch_size) + 1) * epoch_idx

        for train_step in tqdm(
            range(1, len(train_dataiter) + 1),
            desc="Epoch {:03d}: ".format(epoch_idx),
            position=0,
            leave=True,
            disable=not comm.is_main_process(),
        ):
            batch = next(train_dataiter)
            data = batch["data"].cuda()
            target = batch["label"].cuda()

            # forward
            logits_closed_set, logit_all = self.net(data, return_pred_extended=True)
            mask = target < self.config.num_closed_set
            loss_all = F.cross_entropy(logit_all, target)
            if len(target[mask]) > 0:
                loss_closed_set = F.cross_entropy(logits_closed_set[mask], target[mask])
                loss = (loss_closed_set + loss_all) / 2
            else:
                loss = loss_all

            # backward
            total_steps += 1
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()
            self.scheduler.step_update(num_updates=total_steps)

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # comm.synchronize()

        metrics = {}
        metrics["epoch_idx"] = epoch_idx
        metrics["loss"] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced
