import torch
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.utils.config import setup_config
from openood.networks import get_network
from evaluator import BioEvaluator
import os
import pandas as pd

# init
config = setup_config()
dataset_name = config.dataset.name

device = "cuda" if torch.cuda.is_available() else "cpu"

id_loader_dict = get_dataloader(config)
ood_loader_dict = get_ood_dataloader(config)
loader_dict = {"id": id_loader_dict, "ood": ood_loader_dict}
net = get_network(config.network)
checkpoint = torch.load(
    config.network.checkpoint,
    weights_only=True,
)
weights = checkpoint
net.load_state_dict(weights)
net.eval()
net.cuda()

save_arrays = config.save_arrays
evaluator = BioEvaluator(
    net,
    config=config,
    dataloader_dict=loader_dict,
    postprocessor_name=config.postprocessor.name,
    save_arrays=save_arrays,
)

results = evaluator.eval_ood()


dataset_name = config.dataset.name
save_dir = f"output/{dataset_name}/{config.trainer.name}"
os.makedirs(save_dir, exist_ok=True)
csv_path = os.path.join(save_dir, f"{config.postprocessor.name}.csv")
results = results.round(4)
results.to_csv(csv_path, index=False)
