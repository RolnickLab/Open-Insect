# Open-Insect

## Datasets

The metadata for the Open-Insect dataset can be downloaded from [huggingface](https://huggingface.co/datasets/anonymous987654356789/open-insect).

You can run 

```
bash download.sh
```

To download the images and generate metadata for training:

- Modify `download_dir` to change the folder to save the downloaded dataset.
- The `resize_size` is the smaller edge of the image after resizing. Change `resize_size` accordingly.
- If you do not want to resize the images, simply delete `--resize_size 224` from the command. 

Once downloading finishes, 
- images will be saved under `<download_dir>/images`.
- Metadata for training and evaluation will be saved under `<dowload_dir>/metadata`. 
- Change `data_dir`, `imglist_pth`, and `pre_size` in the configs in `configs/datasets` accordingly before training or evaluation.
- You can find the configurations under `configs`.

## Requirements


Run the following commands to install dependencies.

```
conda create -n oi_env python=3.10

conda activate oi_env

pip install -e .

pip install libmr
```

## Training
For training, modify the configs in `scripts/train.sh` and run 

```
bash scripts/train.sh
```

## Evaluation

For evaluation, modify the configs in `scripts/eval.sh` and run 

```
bash scripts/eval.sh
```


## Examples
Here are some minimal examples to test this codebase. First, activate the virtual environment by

```
conda activate oi_env
```
then download the pretrained weights by running

```
python download_pretrained_weights.py
```

The model should be saved under `weights/c-america_resnet50_baseline.pth`.

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
Run 
```
bash scripts/examples/eval.sh
```
The output will be saved as `output/open-insect-example/base/msp.csv`. You can compare the output with `output/open-insect-example/base/msp_expected.csv`.

## Acknowledgement

This codebase is built using [OpenOOD](https://github.com/Jingkang50/OpenOOD/tree/main). We sincerely appreciate their efforts in making this valuable resource publicly available.