from openood.preprocessors import get_preprocessor
from openood.datasets import get_dataloader
from openood.utils.config import setup_config
from openood.networks import get_network
from openood.trainers import get_trainer
from openood.recorders import get_recorder
from openood.evaluators import get_evaluator
from datasets.dataloader import get_id_ood_dataloader_webdataset, get_id_ood_dataloader_folder
from openood.datasets import get_dataloader
from openood.preprocessors import get_preprocessor
import torch
from openood.trainers.lr_scheduler import cosine_annealing
import os
from pathlib import Path
import checkpoint_helper
import wandb
from pipeline_init import get_scheduler, get_optimizer


# init
config = setup_config()
device = "cuda" if torch.cuda.is_available() else "cpu"

# getting environment variables
SCRATCH = os.environ["SCRATCH"]
try:
    SLURM_TMPDIR = Path(os.environ["SLURM_TMPDIR"])
except:
    SLURM_TMPDIR = None
SLURM_JOBID = os.environ["SLURM_JOB_ID"]
# load weights from checkpoint
# get dataloaders
loader_dict = get_dataloader(config)
train_loader = loader_dict["train"]
val_loader = loader_dict["val"]

# initialize
start_epoch, total_steps, best_acc = 0, 0, 0.0
net = get_network(config.network)
optimizer = get_optimizer(net, config)
scheduler = get_scheduler(optimizer, config, train_loader)
wandb_run_id = None
run_dir = Path(os.path.join(SCRATCH, "test"))
checkpoint_dir = run_dir / SLURM_JOBID / "checkpoints"

# Try to resume from a checkpoint, if one exists.
if checkpoint_dir:
    checkpoint: checkpoint_helper.RunState | None = checkpoint_helper.load_checkpoint(
        checkpoint_dir, map_location=device
    )
    if checkpoint:
        wandb_run_id = checkpoint["wandb_run_id"]
        start_epoch = checkpoint["epoch"]  # +1 to start at the next epoch.
        total_steps = checkpoint["total_steps"]
        best_acc = checkpoint["best_acc"]
        net.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        print(
            f"Resuming training at epoch {start_epoch} (best_acc={best_acc:.2%}).",
            flush=True,
        )
    else:
        print(
            f"No checkpoints found in {checkpoint_dir}. Training from scratch.",
            flush=True,
        )


if wandb_run_id == None:
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.name + "_" + SLURM_JOBID,
        resume="allow",
        config=config,
    )
else:
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.name + "_" + SLURM_JOBID,
        id=wandb_run_id,
        resume="must",
        config=config,
    )


trainer = get_trainer(net, train_loader, val_loader, config, optimizer, scheduler)
evaluator = get_evaluator(config)

# add checkpointing
for epoch_idx in range(start_epoch, config.optimizer.num_epochs + 1):
    epoch_metrics, epoch_metrics_formated = {}, {}
    net, train_metrics = trainer.train_epoch(epoch_idx)
    epoch_metrics["train/loss"] = train_metrics["loss"]
    test_metrics = evaluator.eval_acc(net, val_loader, epoch_idx=epoch_idx)

    epoch_metrics["val/loss"] = test_metrics["loss"]
    epoch_metrics["val/acc"] = test_metrics["acc"]

    val_accuracy = test_metrics["acc"]
    is_best = val_accuracy > best_acc
    best_acc = max(val_accuracy, best_acc)
    if checkpoint_dir:
        if torch.cuda.device_count() > 1:
            model_state_dict = net.module.state_dict()
        else:
            model_state_dict = net.state_dict()
        checkpoint_helper.save_checkpoint(
            checkpoint_dir,
            is_best,
            checkpoint_helper.RunState(
                wandb_run_id=wandb.run.id,
                epoch=epoch_idx + 1,
                total_steps=total_steps,
                model_state=model_state_dict,
                optimizer_state=trainer.optimizer.state_dict(),
                scheduler_state=trainer.scheduler.state_dict(),
                best_acc=best_acc,
            ),
        )

    wandb.log({**epoch_metrics})


print("Finished Training", flush=True)
