# Open-Insect

## Datasets

The metadata for the Open-Insect dataset can be downloaded from [huggingface](https://huggingface.co/datasets/anonymous987654356789/open-insect). After downloading the metadata, run `download.sh` to download the images and generate metadata for training.

You can modify `download_dir` to change the folder to save the downloaded dataset. You can change `resize_size` to change the size of which the image will be resized to. The `resize_size` is the smaller edge of the image after resizing. If you do not want to resize the images, simply delete `--resize_size 224` from the command. 

Images will be saved under `<download_dir>/images`.
Metadata for training and evaluation will be saved under `<dowload_dir>/metadata`. Change `data_dir`, `imglist_pth`, and `pre_size` in the configs in `configs/datasets` accordingly. 

## Virtual environment

Create a virtual environment with `python3.10`. Activate the environment and install the packages with the following commands. 
```
pip install -e .

pip install libmr
```

## Running experiments


For training, modify `scripts/train.sh` and run `bash scripts/train.sh`.

For evaluation, modify `scripts/eval.sh` and run `bash scripts/eval.sh`.

You can find the configurations under `configs`.

## Examples
Here is a minimal example to test this codebase. First, activate the virtual environment. 

### Training

To test training methods that do no require auxiliary data, run 
```
bash scripts/examples/train.sh
```

To test training methods that require auxiliary data, run 
```
bash scripts/examples/train_with_aux_data.sh
```

The training and validation accuracy are expected to be 0 after 2 epochs as there is only 1 image per speices in the training set, and the model is trained from scratch.


### Evaluation
First, download the model weights from [huggingface](https://huggingface.co/anonymous987654356789/open-insect-test-model/blob/main/c-america_resnet50_baseline.pth) and place it under `weights`.

Then run 
```
bash scripts/examples/eval.sh
```
The output will be saved as `output/open-insect-example/base/msp.csv`. You can compare the output with `output/open-insect-example/base/msp_expected.csv`.
## Acknowledgement

This codebase is built using [OpenOOD](https://github.com/Jingkang50/OpenOOD/tree/main). We sincerely appreciate their efforts in making this valuable resource publicly available.