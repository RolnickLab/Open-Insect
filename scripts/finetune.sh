#!/bin/bash

REGION=$1
METHOD=$2
NETWORK=$3
OUTPUT_DIR=$4
CHECKPOINT=$5

if [ "$METHOD" = "udg" ]; then
    DATASET_CLASS="UDGDataset"
else
    DATASET_CLASS="ImglistDataset"
fi


if [ "$REGION" = "ne-america" ]; then
    if [ "$METHOD" = "extended" ]; then
         NUM_CLASSES=10497
    elif [ "$METHOD" = "novel_branch" ]; then
         NUM_CLASSES=2498
    else
         NUM_CLASSES=2497
    fi
elif [ "$REGION" = "w-europe" ]; then
    if [ "$METHOD" = "extended" ]; then
         NUM_CLASSES=10603
    elif [ "$METHOD" = "novel_branch" ]; then
         NUM_CLASSES=2604
    else
         NUM_CLASSES=2603
    fi
elif [ "$REGION" = "c-america" ]; then
    if [ "$METHOD" = "extended" ]; then
         NUM_CLASSES=4636
    elif [ "$METHOD" = "novel_branch" ]; then
         NUM_CLASSES=637 # ID classes + 1
    else
         NUM_CLASSES=636 # ID classes
    fi
else
    NUM_CLASSES=-1
fi


# for finetuning
python scripts/main.py \
 --config configs/datasets/$REGION.yml \
   configs/datasets/${REGION}_oe.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/networks/resnet50.yml \
    configs/networks/${NETWORK}.yml \
    configs/pipelines/train/baseline.yml \
    configs/pipelines/train/train_$METHOD.yml \
     --network.pretrained True \
     --network.checkpoint $CHECKPOINT \
     --optimizer.num_epochs 1 \
     --optimizer.warmup_epochs 0 \
     --optimizer.lr 0.001 \
     --trainer.name $METHOD \
     --dataset.name $REGION \
     --dataset.train.dataset_class $DATASET_CLASS \
     --dataset.oe.dataset_class $DATASET_CLASS  \
     --dataset.num_classes $NUM_CLASSES \
     --dataset.train.batch_size 32 \
     --num_gpus 1 --num_workers 2 \
     --merge_option merge \
     --seed 0 \
     --output_dir $OUTPUT_DIR \
     --exp_name ${REGION}/${METHOD}"/finetune/s'@{seed}'" \
   