Note: This codebase has only been tested on Debian-based Linux systems. CUDA is required to install the dependencies and run experiments.  

# The Open-Insect Dataset

## Open-Insect
The Open-Insect dataset with GBIF images is publicly avaiable at [Open-Insect](https://huggingface.co/datasets/yuyan-chen/open-insect) on hugggingface.

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


## Open-Insect-BCI

The C-America O-BCI dataset is hosted separately at [Open-Insect-BCI](https://huggingface.co/datasets/yuyan-chen/open-insect-bci) on huggingface.

Run

```
python download_bci.py --download_dir .
```
to download the BCI dataset to the current directory, or change the `download_dir` accordingly.

- Images will be saved under `<download_dir>/images/bci`.
- Metadata will be saved as `<dowload_dir>/metadata/c-america/test_ood_bci.txt`. 

# Requirements


Run the following commands to install dependencies.

```
conda create -n oi_env python=3.10

conda activate oi_env

pip install -e .

pip install libmr
```

The default batch size is 512 and the number of works is 16. With this setting, models can be trained with 1 RTX800 GPU with 48 GB memory, 16 CPUs (and 16 workers), and 100 GB CPU memory in total. 


# Arguments

| Argument             | Description                                                               | Possible Values / Examples                                                                      |
| -------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `REGION`        | Specifies the geographical region of the dataset to evaluate.             | `ne-america` (Northeastern America), `w-europe` (Western Europe), `c-america` (Central America) |
| `METHOD`         | The training method.                            | See the list `METHOD` Options below.                                                        |
| `POSTHOC_METHOD` | The post-hoc open-set detection method applied to the trained classifier. | See the list  `POSTHOC_METHOD` Options below.                                                |
| `NETWORK`        | The backbone network used in the model.                                   | See the list `NETWORK` Options below.                                                       |
| `CHECKPOINT_DIR`     | Path to the directory to save the trained model checkpoints.           | Example: `$HOME/weights`                                |



`METHOD` Options (See Table 2 in the paper for more details of the methods)
- `basics` - the basic classifier trained with Cross Entropy loss with only the closed-set 
-  `conf_branch` - ConfBranch
-  `logitnorm` - LogitNorm
-  `godin` - GODIN
-  `rotpred`- RotPred
-   `oe` - OE
-    `udg` - UDG
-    `mixoe` - MixOE
-    `energy` - Energy
-    `extended` - Extended
-    `novel_branch`- NovelBranch

---
`POSTHOC_METHOD` Options (See Table 2 in the paper for more details of the post-hoc methods)

- Generic post-hoc methods
    - `openmax` — OpenMax
    - `msp` — MSP
    - `temperature_scaling` — TempScale
    - `odin` — ODIN 
    - `mds` — MDS
    - `mds_ensemble` — MDSEns
    - `rmds` — RMDS
    - `gram` — Gram 
    - `ebo` — EBO
    - `gradnorm` — GradNorm
    - `react` - ReAct
    - `mls` — MLS
    - `klm` — KLM
    - `vim` — VIM
    - `knn` — k-Nearest Neighbor in feature space  
    - `dice` - DICE
    - `rankfeat` - RankFeat
    - `ash` — ASH 
    - `she` — SHE
    - `neco` — NECO
    - `fdbd` — FDBD
    - `rp_msp`, `rp_odin`, `rp_ebo`, `rp_gradnorm` - RP_MSP, RP_ODIN, RP_EBO, RP_GradNorm
    - `nci` — NCI
- Post-hoc methods for a specific training method
    - `conf_branch` - To be used with `METHOD`: `conf_branch`
    - `godin` - To be used with `METHOD`: `godin`
    - `rotpred` - To be used with `METHOD`: `rotpred`
---

`NETWORK` Options 
- `conf_branch` - for `METHOD`: `conf_branch`
- `godin_net` - for `METHOD`: `godin`
- `rot_net` - for `METHOD`: `rotpred`
- `udg_net` - for `METHOD`: `udg`
- `extended_net` - for `METHOD`: `extended` or `novel_branch`
- `resnet50` - for all other methods

# Training
Run the following command to train from scratch. The model checkpoint will be saved in `${CHECKPOINT_DIR}/${REGION}/${METHOD}/train_from_scratch/s${RANDOM_SEED}`.

```
bash scripts/train.sh REGION METHOD NETWORK CHECKPOINT_DIR
```

Run the following command to fine-tune the `CHECKPOINT`. The fine-tuned model will be saved in  `CHECKPOINT_DIR`.

```
bash scripts/finetune.sh REGION METHOD NETWORK CHECKPOINT_DIR CHECKPOINT
```

<!-- For training methods that do not require auxilairy data, modify the configs in `scripts/train.sh` and run 

```
bash scripts/train.sh
```
You can find the configurations under `configs`.

For training methods that require auxilairy data, modify the configs in `scripts/train_with_aux_data.sh` and run 

```
bash scripts/train.sh
```
You can find the configurations under `configs`. -->


# Evaluation


## Evaluating ARPL

```
bash scripts/eval_arpl.sh REGION arpl msp arpl_net CHECKPOINT_DIR
```

## Evaluating OpenGAN


```
bash scripts/eval_opengan.sh REGION opengan opengan resnet50 CHECKPOINT_DIR
```

## Evaluating other methods
For all other methods, run 

```
bash scripts/eval.sh REGION METHOD POSTHOC_METHOD NETWORK CHECKPOINT_DIR
```


For example, to evaluate the basic classifier for Central America with MSP using the checkpoint saved under `$HOME/weights`, run


```
bash scripts/eval.sh c-america basics msp resnet50 $HOME/weights
```
See `scripts/test_eval_script.sh` for more examples. 

## Possible errors
- OpenMax: This method requires predictions of the test set to cover all training species. Otherwise, the following error will occur: `RuntimeError: torch.cat(): expected a non-empty list of Tensors`.

# Pre-trained checkpoints
Checkpoints can be downloaded from https://huggingface.co/yuyan-chen/open-insect-model-weights or by running 

```
python download_pretrained_weights.py --weight_dir WEIGHT_DIR
```







# Examples
Here are some minimal examples to test this codebase. First, activate the virtual environment by

```
conda activate oi_env
```
then download the pretrained weights by running

```
python download_pretrained_weights.py
```

The model should be saved under `weights/c-america_resnet50_baseline.pth`.

## Training

To test training methods that do no require auxiliary data, run 
```
bash scripts/examples/train.sh
```

To test training methods that require auxiliary data, run 
```
bash scripts/examples/train_with_aux_data.sh
```

The training and validation accuracy are expected to be 0 after 2 epochs as there is only 1 image per speices in the training set, and the model is trained from scratch.


## Evaluation
Run 
```
bash scripts/examples/eval.sh
```
The outputs will be saved in `results/open-insect-example/base/msp/ood.csv`. You can compare the outputs with `example/ood.csv`.

# Acknowledgement

This codebase is built using [OpenOOD](https://github.com/Jingkang50/OpenOOD/tree/main). We sincerely appreciate their efforts in making this valuable resource publicly available.