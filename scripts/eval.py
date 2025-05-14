import torch
import argparse
import yaml
import time
import wandb
from src.OpenOOD.openood.networks import ResNet50
from src.OpenOOD.openood.networks import get_network
from src.OpenOOD.openood.utils.config import setup_config
from src.evaluator import BioEvaluator
from src.datasets.dataloader import get_id_ood_dataloader_folder
from src.OpenOOD.openood.datasets import get_dataloader
import pathlib

parser = argparse.ArgumentParser(prog="BioOOD")
parser.add_argument("--postprocessor_name")
parser.add_argument("--method_name")
parser.add_argument("--network_name")
parser.add_argument("--slurm_jobid")
parser.add_argument("--dataset_name")
parser.add_argument("--checkpoint_path")
parser.add_argument("--dataset_config")
parser.add_argument("--postprocessor_config_path")
parser.add_argument("--backbone_name")
parser.add_argument("--wandb_project")
parser.add_argument("--wandb_entity")
parser.add_argument("--num_clusters", type=int)  # for udg
args = parser.parse_args()
num_classes_dict = {"ne-america": 2497, "w-europe": 2603, "c-america": 636}

# # init
# config = setup_config()
# device = "cuda" if torch.cuda.is_available() else "cpu"



class NetworkConfig:
    def __init__(self, name, num_classes):
        self.name = name
        self.num_classes = num_classes
        self.pretrained = False  # load weigts after initializing the network
        self.num_gpus = 1
        self.backbone = None
        self.num_clusters = None
        self.wandb_project = None
        self.wandb_entity = None


device = "cuda" if torch.cuda.is_available() else "cpu"
loader_dict = get_id_ood_dataloader_folder(args.dataset_config)
network_config = NetworkConfig(args.network_name, num_classes_dict[args.dataset_name])
if args.backbone_name:
    backbone_config = NetworkConfig(args.backbone_name, num_classes_dict[args.dataset_name])
    network_config.backbone = backbone_config
if args.num_clusters:
    print(args.num_clusters)
    network_config.num_clusters = args.num_clusters


if args.wandb_project:
    network_config.wandb_project = args.wandb_project
    network_config.wandb_entity = args.wandb_entity
    wandb.init(
        project=network_config.wandb_project,
        entity=network_config.wandb_entity,
        name=f"{args.dataset_name}_{args.method_name}_{args.postprocessor_name}",
        resume="allow",
        config=network_config,
    )


net = get_network(network_config)
checkpoint = torch.load(
    args.checkpoint_path,
    weights_only=False,
)
weights = checkpoint["model_state"]
net.load_state_dict(weights)
net.eval()
net.cuda()

evaluator = BioEvaluator(
    net,
    id_name=args.dataset_name,
    data_config_path=args.dataset_config,
    config_root=args.postprocessor_config_path,
    dataloader_dict=loader_dict,
    postprocessor_name=args.postprocessor_name,
)

results = evaluator.eval_ood()

save_dir = f"output/{args.dataset_name}/{args.method_name}"
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
# save the result
results.to_csv(f"{save_dir}/{args.postprocessor_name}.csv", index=False)
if args.wandb_project:
    wandb.log({"result": results})
