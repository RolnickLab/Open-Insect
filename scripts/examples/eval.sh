#!/bin/bash

python scripts/eval_trained.py \
    --config configs/datasets/example.yml  \
      configs/datasets/example_ood.yml \
      configs/preprocessors/base_preprocessor.yml \
      configs/networks/resnet50.yml \
      configs/pipelines/test/test_ood.yml \
      configs/postprocessors/msp.yml \
    --network.checkpoint weights/c-america_resnet50_baseline.pth \
    --trainer.name base \
    --num_gpus 1 --num_workers 1 \
    --merge_option merge \
    --seed 0 
