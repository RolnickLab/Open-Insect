import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config

from .base_trainer import BaseTrainer


class EnergyTrainer(BaseTrainer):
    def __init__(
        self,
        net: nn.Module,
        train_loader: DataLoader,
        train_unlabeled_loader: DataLoader,
        config: Config,
    ) -> None:
        super().__init__(net, train_loader, config)
        self.train_unlabeled_loader = train_unlabeled_loader
        self.lambda_energy = config.trainer.lambda_energy
        self.m_in, self.m_out = self.setup()

        print(self.m_in, self.m_out)

    @torch.no_grad()
    def setup(self):
        self.net.eval()  # enter train mode
        train_dataiter = iter(self.train_loader)
        unlabeled_dataiter = iter(self.train_unlabeled_loader)

        # m_in
        total_energy, total_count = 0, 0
        for _ in tqdm(
            range(1, len(train_dataiter) + 1),
            desc="Computeing m_in",
            position=0,
            leave=True,
            disable=not comm.is_main_process(),
        ):
            batch = next(train_dataiter)
            data = batch["data"].cuda()
            logits = self.net(data)
            energy = -torch.logsumexp(logits, dim=1)
            total_energy += energy.sum()
            total_count += len(energy)

        m_in = total_energy / total_count

        # m_out
        total_energy, total_count = 0, 0
        for _ in tqdm(
            range(1, len(unlabeled_dataiter) + 1),
            desc="Computeing m_out",
            position=0,
            leave=True,
            disable=not comm.is_main_process(),
        ):
            batch = next(unlabeled_dataiter)
            data = batch["data"].cuda()
            logits = self.net(data)
            energy = -torch.logsumexp(logits, dim=1)
            total_energy += energy.sum()
            total_count += len(energy)
        m_out = total_energy / total_count

        return m_in, m_out

    def train_epoch(self, epoch_idx):
        self.net.train()  # enter train mode

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)
        total_steps = (int(self.config.dataset.train.len / self.config.dataset.train.batch_size) + 1) * epoch_idx

        if self.train_unlabeled_loader:
            unlabeled_dataiter = iter(self.train_unlabeled_loader)

        for train_step in tqdm(
            range(1, len(train_dataiter) + 1),
            desc="Epoch {:03d}: ".format(epoch_idx),
            position=0,
            leave=True,
            disable=not comm.is_main_process(),
        ):
            batch = next(train_dataiter)

            try:
                unlabeled_batch = next(unlabeled_dataiter)
            except StopIteration:
                unlabeled_dataiter = iter(self.train_unlabeled_loader)
                unlabeled_batch = next(unlabeled_dataiter)

            data = torch.cat((batch["data"], unlabeled_batch["data"])).cuda()
            batch_size = batch["data"].size(0)

            # forward
            logits_classifier = self.net(data)
            logits_in, logits_out = logits_classifier[:batch_size], logits_classifier[batch_size:]
            loss = F.cross_entropy(logits_in, batch["label"].cuda())

            Ec_out = -torch.logsumexp(logits_out, dim=1)
            Ec_in = -torch.logsumexp(logits_in, dim=1)
            loss += self.lambda_energy * (
                torch.pow(F.relu(Ec_in - self.m_in), 2).mean() + torch.pow(F.relu(self.m_out - Ec_out), 2).mean()
            )

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

        metrics = {}
        metrics["epoch_idx"] = epoch_idx
        metrics["loss"] = self.save_metrics(loss_avg)

        return self.net, metrics
