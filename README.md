# Open-Insect

This codebase has only been tested on Debian-based Linux systems. CUDA is required. 

## Datasets

The Open-Insect dataset is publicly avaiable at [huggingface](https://huggingface.co/datasets/anonymous987654356789/open-insect).


To download the images and generate metadata for training:

- Change `download_dir` to the directory where you want the downloaded dataset to be saved.
- The `resize_size` is the smaller edge of the image after resizing. Change `resize_size` accordingly. The default value is 224. 
- If you do not want to resize the images, simply delete `--resize_size 224` from the command. Without resizing, the downloaded images will require approximately 6TB of storage.
- Run 
    ```
    bash download.sh
    ```

Once downloading finishes, 
- Images will be saved under `<download_dir>/images`.
- Metadata for training and evaluation will be saved under `<dowload_dir>/metadata`. 
- Change `data_dir`, `imglist_pth`, and `pre_size` in the configs under `configs/datasets` accordingly before training or evaluation.


## Requirements


Run the following commands to install dependencies.

```
conda create -n oi_env python=3.10

conda activate oi_env

pip install -e .

pip install libmr
```

## Experiments
The default batch size is 512 and the number of works is 16. With this setting, models can be trained with 1 RTX800 GPU with 48 GB memory, 16 CPUs (and 16 workers), and 100 GB CPU memory in total. 

### Training
For training, modify the configs in `scripts/train.sh` and run 

```
bash scripts/train.sh
```
You can find the configurations under `configs`.



### Evaluation

For evaluation, modify the configs in `scripts/eval.sh` and run 

```
bash scripts/eval.sh
```
You can find the configurations under `configs`.


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