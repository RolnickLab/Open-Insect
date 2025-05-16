from datasets import load_dataset
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--download_dir", type=str, help="Path to save the downloaded dataset"
)
args = parser.parse_args()

if __name__ == "__main__":

    ds = load_dataset("anonymous987654356789/open-insect-bci", split="test")

    image_dir = os.path.join(args.download_dir, "images", "bci")
    metadata_dir = os.path.join(args.download_dir, "metadata", "c-america")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    with open(os.path.join(metadata_dir, "test_ood_bci.txt"), "w") as f:
        for image, img_path in zip(ds["image"], ds["image_path"]):
            image.save(os.path.join(image_dir, img_path))
            f.write(f"bci/{img_path} -1\n")
