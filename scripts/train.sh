#!/bin/bash


# for training from scratch 
python scripts/main.py \
 --config configs/datasets/<dataset config>.yml \
    configs/preprocessors/<preprocessor config>.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/train/baseline.yml \
    configs/pipelines/train/<the training pipeline>.yml \
    --network.pretrained False \
    --optimizer.num_epochs 120 \
    --optimizer.warmup_epochs 6 \
    --optimizer.lr 0.01 \
    --dataset.train.batch_size 512 \
    --num_gpus 1 --num_workers 16 \
    --merge_option merge \
    --seed 0 
   

# for finetuning
python scripts/main.py \
 --config configs/datasets/<dataset config>.yml \
    configs/preprocessors/<preprocessor config>.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/train/baseline.yml \
    configs/pipelines/train/<the training pipeline>.yml \
   --network.pretrained True \
    --network.checkpoint <path to the checkpoint>  \
    --optimizer.num_epochs 30 \
    --optimizer.warmup_epochs 2 \
    --optimizer.lr 0.001 \
    --dataset.train.batch_size 512 \
    --num_gpus 1 --num_workers 16 \
    --merge_option merge \
    --seed 0 
   