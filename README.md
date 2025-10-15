# Open-Insect 

This codebase has only been tested on Debian-based Linux systems. CUDA is required to install the dependencies and run experiments.  


## Datasets

### Open-Insect
The Open-Insect dataset with GBIF images is publicly avaiable at [open-insect](https://huggingface.co/datasets/anonymous987654356789/open-insect) on hugggingface.

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
- Metadata for training and evaluation of each region will be saved as
    ```
    <download_dir>/metadata/<region>
    │   ├── test_id.csv
    │   ├── test_ood_local.csv
    │   ├── test_ood_non-local.csv
    │   ├── test_ood_non-moth.csv
    │   ├── train_aux.csv
    │   ├── train_id.csv
    │   ├── val_id.csv
    │   └── val_ood.csv
    ```
- Change `data_dir`, `imglist_pth`, and `pre_size` in the configs under `configs/datasets` accordingly before training or evaluation.


### Open-Insect-BCI

The C-America OOD-BCI dataset is hosted separately at [open-insect-bci](https://huggingface.co/datasets/anonymous987654356789/open-insect-bci) on huggingface.

Run

```
python download_bci.py --download_dir .
```
to download the BCI dataset to the current directory, or change the `download_dir` accordingly.

- Images will be saved under `<download_dir>/images/bci`.
- Metadata will be saved as `<dowload_dir>/metadata/c-america/test_ood_bci.txt`. 

## Requirements


Run the following commands to install dependencies.

```
conda create -n oi_env python=3.10

conda activate oi_env

pip install -e .

pip install libmr
```

The default batch size is 512 and the number of works is 16. With this setting, models can be trained with 1 RTX800 GPU with 48 GB memory, 16 CPUs (and 16 workers), and 100 GB CPU memory in total. 

## Training
For training methods that do not require auxilairy data, modify the configs in `scripts/train.sh` and run 

```
bash scripts/train.sh
```
You can find the configurations under `configs`.

For training methods that require auxilairy data, modify the configs in `scripts/train_with_aux_data.sh` and run 

```
bash scripts/train.sh
```
You can find the configurations under `configs`.


## Evaluation

To evaluate ARPL, run 

```
bash scripts/eval_arpl.sh REGION METHOD POSTHOC_METHOD NETWORK WEIGHT_DIR
```

To evaluate OpenGAN, run


```
bash scripts/eval_gan.sh REGION METHOD POSTHOC_METHOD NETWORK WEIGHT_DIR
```

For all other methods, run 

```
bash scripts/eval.sh REGION METHOD POSTHOC_METHOD NETWORK WEIGHT_DIR
```

For example, to evaluate the basic classifier for Central America with MSP using the checkpoint saved under `$HOME/weights`, run


```
bash scripts/eval.sh c-america basics msp resnet50 $HOME/weights
```


See `scripts/test_eval_script.sh` for more examples. 

### Possible errors
- OpenMax: This method requires predictions of the test set to cover all training species. Otherwise, the following error will occur: `RuntimeError: torch.cat(): expected a non-empty list of Tensors`.


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