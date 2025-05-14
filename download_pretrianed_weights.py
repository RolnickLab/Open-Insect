from huggingface_hub import hf_hub_download
import shutil
import os

path = hf_hub_download(
    repo_id="anonymous987654356789/open-insect-test-model",
    filename="c-america_resnet50_baseline.pth",
)
os.makedirs("weights", exist_ok=True)
shutil.move(path, "weights")
