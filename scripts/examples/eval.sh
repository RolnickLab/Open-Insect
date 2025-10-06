#!/bin/bash

python scripts/eval_trained.py \
    --config configs/datasets/example.yml  \
      configs/datasets/example_ood.yml \
      configs/preprocessors/base_preprocessor.yml \
      configs/networks/resnet50.yml \
      configs/networks/conf_branch.yml \
      configs/pipelines/train/baseline.yml \
      configs/pipelines/test/test_ood.yml \
      configs/postprocessors/msp.yml \
    --network.pretrained True \
    --network.checkpoint /network/scratch/y/yuyan.chen/open_insect_camera_ready/weights/conf_branch_c-america.pth \
    --network.backbone.name resnet50 \
    --num_gpus 1 --num_workers 1 \
    --dataset.train.batch_size 32 \
    --dataset.val.batch_size 32 \
    --dataset.test.batch_size 32 \
    --merge_option merge \
    --seed 0 


#  --network.checkpoint weights/c-america_resnet50_baseline.pth \