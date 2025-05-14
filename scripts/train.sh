#!/bin/bash
module load python/3.10

source ~/<env>/bin/activate

PYTHONPATH=$PYTHONPATH
PYTHONPATH=src:$PYTHONPATH
PYTHONPATH=src/OpenOOD:$PYTHONPATH

export PYTHONPATH

python scripts/main.py \
 --config configs/datasets/<dataset config>.yml \
    configs/preprocessors/<preprocessor config>.yml \
    configs/networks/<network config>.yml \
    configs/pipelines/train/baseline.yml \
    --optimizer.num_epochs 120 \
    --optimizer.warmup_epochs 6 \
    --optimizer.lr 0.01 \
    --run_dir <directory to save checkpoints> \
    --dataset.train.batch_size 512 \
    --num_gpus 1 --num_workers 16 \
    --merge_option merge \
    --seed 0 