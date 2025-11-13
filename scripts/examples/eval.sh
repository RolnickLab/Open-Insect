#!/bin/bash

python scripts/main.py \
    --config configs/datasets/example_c-america.yml  \
      configs/datasets/example_c-america_ood_test.yml \
      configs/networks/resnet50.yml \
      configs/preprocessors/base_preprocessor.yml \
      configs/pipelines/train/baseline.yml \
      configs/pipelines/test/test_ood.yml \
      configs/postprocessors/msp.yml \
    --network.pretrained True \
    --network.checkpoint weights/basics_c-america.pth \
    --num_gpus 1 --num_workers 1 \
    --dataset.train.batch_size 32 \
    --dataset.val.batch_size 32 \
    --dataset.test.batch_size 32 \
    --merge_option merge \
    --seed 0 

