#!/bin/bash

python scripts/main.py \
 --config configs/datasets/example_c-america.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/train/baseline.yml \
    --optimizer.num_epochs 2 \
    --optimizer.warmup_epochs 1 \
    --optimizer.lr 0.01 \
   --dataset.train.batch_size 32 \
    --dataset.val.batch_size 32 \
    --dataset.test.batch_size 32 \
    --num_gpus 1 --num_workers 1 \
    --merge_option merge \
    --run_dir example \
    --seed 0