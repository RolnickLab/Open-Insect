#!/bin/bash
#SBATCH --job-name=arpl
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --partition=long             
#SBATCH --cpus-per-task=8      
#SBATCH --gres=gpu:rtx8000:1   

REGION=$1
METHOD=$2
POSTHOC_METHOD=$3
NETWORK=$4
WEIGHT_DIR=$5


python scripts/main.py \
    --config configs/datasets/$REGION.yml  \
    configs/datasets/${REGION}_ood_test.yml \
    configs/networks/resnet50.yml \
    configs/networks/${NETWORK}.yml \
    configs/pipelines/train/baseline.yml \
    configs/pipelines/train/train_arpl.yml \
    configs/pipelines/test/test_arpl.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/$POSTHOC_METHOD.yml \
    --trainer.name arpl \
    --dataset.name $REGION \
    --network.pretrained True \
    --network.backbone.name resnet50 \
    --network.feat_extract_network.name resnet50 \
    --network.weight_dir $WEIGHT_DIR \
    --num_gpus 1 --num_workers 8 \
    --dataset.train.batch_size 512 \
    --merge_option merge \
    --seed 0 \
    --output_dir $SCRATCH/open_insect_test/output