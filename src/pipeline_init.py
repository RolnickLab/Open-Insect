import torch
from openood.trainers.lr_scheduler import cosine_annealing
import wandb
import signal
from types import FrameType
from timm.scheduler import CosineLRScheduler


def get_optimizer(
    net,
    config,
):

    optimizer = torch.optim.SGD(
        net.parameters(),
        config.optimizer.lr,
        momentum=config.optimizer.momentum,
        weight_decay=config.optimizer.weight_decay,
        nesterov=True,
    )

    return optimizer


def get_scheduler(optimizer, config, train_loader):
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            config.optimizer.num_epochs * len(train_loader),
            1,
            1e-6 / config.optimizer.lr,
        ),
    )

    return scheduler


def signal_handler(signum: int, frame: FrameType | None):
    """Called before the job gets pre-empted or reaches the time-limit.

    This should run quickly. Performing a full checkpoint here mid-epoch is not recommended.
    """
    signal_enum = signal.Signals(signum)
    print(f"Job received a {signal_enum.name} signal!", flush=True)
    if wandb.run:
        wandb.mark_preempting()


def print_metrics(metrics):
    return " - ".join([f"{k}: {v}" for k, v in metrics.items()])
