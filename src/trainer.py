import openood.trainers
import torch
import torch.nn.functional as F


class NewBaseTrainer(openood.trainers.base_trainer.BaseTrainer):
    def __init__(
        self,

    ):
        print("hello world")

    def print_message(self):
        print("created new base trainer")

    def train_epoch(self, epoch_idx):
        self.net.train()

        loss_avg = 0.0

        for batch in self.train_loader:
            data = batch["data"].cuda()
            target = batch["label"].cuda()

            # forward
            logits_classifier = self.net(data)
            loss = F.cross_entropy(logits_classifier, target)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # comm.synchronize()

        metrics = {}
        metrics["epoch_idx"] = epoch_idx
        metrics["loss"] = self.save_metrics(loss_avg)

        return self.net, metrics


openood.trainers.base_trainer.BaseTrainer = NewBaseTrainer


from openood.trainers.logitnorm_trainer import LogitNormTrainer

trainer = NewBaseTrainer()
trainer.print_message()
