#!/bin/bash

REGION=$1
METHOD=$2
POSTHOC_METHOD=$3
NETWORK=$4
WEIGHT_DIR=$5


python scripts/main.py \
    --config configs/datasets/example_$REGION.yml  \
    configs/datasets/example_${REGION}_ood_test.yml \
    configs/networks/resnet50.yml \
    configs/networks/${NETWORK}.yml \
    configs/pipelines/train/baseline.yml \
    configs/pipelines/train/train_$METHOD.yml \
    configs/pipelines/test/test_arpl.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/$POSTHOC_METHOD.yml \
    --trainer.name $METHOD \
    --dataset.name $REGION \
    --network.pretrained True \
    --network.backbone.name resnet50 \
    --network.feat_extract_network.name resnet50 \
    --network.weight_dir $WEIGHT_DIR \
    --num_gpus 1 --num_workers 1 \
    --dataset.train.batch_size 32 \
    --dataset.val.batch_size 32 \
    --dataset.test.batch_size 32 \
    --merge_option merge \
    --seed 0 \
    --output_dir $SCRATCH/open_insect_test/output