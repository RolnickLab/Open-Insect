import shutil
import os
from huggingface_hub import snapshot_download
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--weight_dir", type=str, help="Path to save the downloaded weights"
)
args = parser.parse_args()


# Download all files from the repo
snapshot_path = snapshot_download(
    repo_id="yuyan-chen/open-insect-model-weights",
    allow_patterns="*.pth",
)

pth_files = glob.glob(os.path.join(snapshot_path, "*.pth"))

for path in pth_files:
    shutil.copy(path, os.path.join("weights", os.path.basename(path)))

    os.makedirs(args.weight_dir, exist_ok=True)
    shutil.copy(
        path,
        os.path.join(
            args.weight_dir,
            os.path.basename(path),
        ),
    )
