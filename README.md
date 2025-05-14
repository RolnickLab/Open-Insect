# Open-Insects 

## Datasets

The dataset is on [huggingface](https://huggingface.co/datasets/anonymous987654356789/open-insect).

## Virtual environment

It requires python 3.10. 
```
pip install -e .

pip install libmr
```

## Running experiments


For training, modify `scripts/train.sh` and run `bash scripts/train.sh`.

For evaluation, modify `scripts/eval.sh` and run `bash scripts/eval.sh`.


## Examples

first download the weights from [huggingface](https://huggingface.co/anonymous987654356789/open-insect-test-model/blob/main/c-america_resnet50_baseline.pth) and put it under `weights`

activate the virtual environment and run 
```
bash scripts/examples/eval.sh
```
Check if the output `output/open-insect-example/base/msp.csv` is the same as `output/open-insect-example/base/msp_expected.csv`.
## Acknowledgement

This codebase is built using [OpenOOD](https://github.com/Jingkang50/OpenOOD/tree/main). We sincerely appreciate their efforts in making this valuable resource publicly available.