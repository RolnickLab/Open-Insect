import os
from PIL import Image
import requests
from datasets import load_dataset
import torch
from torchvision import transforms
import argparse


def get_resize_transform(resize_min_size):
    return transforms.Compose(
        [transforms.Resize(resize_min_size), transforms.ToTensor()]
    )


def resize_image(fpath, resize_size, log, current_index):
    resize_transform = get_resize_transform(resize_size)
    try:
        image = Image.open(fpath)
    except Exception as e:
        log.write(f"{current_index} {e}\n")
        return 0
    image = image.convert("RGB")
    image = resize_transform(image)
    image = image.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
    image = image.numpy().transpose(1, 2, 0)
    image = Image.fromarray(image)
    image.save(fpath)

    return 1


def download_images_from_metadata(
    hf_dataset_split,
    download_dir,
    region,
    split,
    resize_size=None,
):
    image_dir = os.path.join(download_dir, "images")
    metadata_dir = os.path.join(download_dir, "metadata", region)
    log_dir = os.path.join(download_dir, "logs", region)

    os.makedirs(image_dir, exist_ok=True)
    os.chmod(image_dir, 0o755)
    os.makedirs(metadata_dir, exist_ok=True)
    os.chmod(metadata_dir, 0o755)
    os.makedirs(log_dir, exist_ok=True)
    os.chmod(log_dir, 0o755)

    metadata_filename = f"{split}.txt"
    metadata_path = os.path.join(metadata_dir, metadata_filename)
    log_path = os.path.join(log_dir, metadata_filename)

    latest_downloaded_index = 0

    if os.path.isfile(log_path):
        with open(log_path, "rb") as log_file:
            try:  # catch OSError in case of a one line file
                log_file.seek(-2, os.SEEK_END)
                while log_file.read(1) != b"\n":
                    log_file.seek(-2, os.SEEK_CUR)
            except OSError:
                log_file.seek(0)
            last_line = log_file.readline().decode()

            latest_downloaded_index = int(last_line.split(" ")[0]) + 1

    with open(metadata_path, "a") as metadata, open(log_path, "a") as log:

        if latest_downloaded_index == 0:
            print("Start downloading...", flush=True)
        elif latest_downloaded_index == len(hf_dataset_split):
            print("Download finished", flush=True)
            return

        else:
            print(f"Resume downloading from image {latest_downloaded_index}")

        subset = hf_dataset_split.select(
            range(latest_downloaded_index, len(hf_dataset_split))
        )

        for i, row in enumerate(subset, start=latest_downloaded_index):
            url = row["identifier"]
            fpath = row["image_path"]
            fpath = os.path.join(image_dir, fpath)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)

            if not os.path.exists(fpath):
                try:
                    response = requests.get(url, timeout=20)
                    with open(fpath, "wb") as img:
                        img.write(response.content)
                except Exception as e:
                    log.write(f"Error downloading {url}: {e}")
                    log.flush()
                    continue

            if resize_size:
                if not resize_image(fpath, resize_size, log, i):
                    continue

            metadata.write(
                f"{row['image_path']} {row['label']}\n"
            )  # only include the image in the metadata if we successfull resize it
            metadata.flush()
            log.write(f"{i}\n")
            log.flush()
            print(i)

        print("Download finished", flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download_dir", type=str, help="Path to save the downloaded dataset"
    )
    parser.add_argument(
        "--resize_size",
        type=int,
        default=None,
        help="Path to save the downloaded dataset",
    )
    parser.add_argument(
        "--region_name",
        type=str,
        default=None,
        help="Path to save the downloaded dataset",
    )
    args = parser.parse_args()

    region_name = args.region_name
    ds = load_dataset("yuyan-chen/open-insect", region_name)

    for split, dataset in ds.items():
        ood_split = split.split("_")[1]
        download_images_from_metadata(
            hf_dataset_split=dataset,
            download_dir=args.download_dir,
            region=region_name,
            split=split,
            resize_size=args.resize_size,
        )
