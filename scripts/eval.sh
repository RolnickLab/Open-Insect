#!/bin/bash
#SBATCH --job-name=post_hoc
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

module load anaconda/3

conda activate oi_env

python scripts/main.py \
    --config configs/datasets/$REGION.yml  \
    configs/datasets/${REGION}_ood_test.yml \
    configs/networks/resnet50.yml \
    configs/networks/${NETWORK}.yml \
    configs/pipelines/train/baseline.yml \
    configs/pipelines/train/train_$METHOD.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/$POSTHOC_METHOD.yml \
    --use_standard_pred True \
    --trainer.name $METHOD \
    --dataset.name $REGION \
    --dataset.num_classes $NUM_CLASSES \
    --dataset.train.dataset_class $DATASET_CLASS \
    --network.pretrained True \
    --network.checkpoint $WEIGHT_DIR/${METHOD}_${REGION}.pth \
    --network.backbone.name resnet50 \
    --num_gpus 1 --num_workers 8 \
    --dataset.train.batch_size 512 \
    --merge_option merge \
    --seed 0 \
    --output_dir $SCRATCH/open_insect_test/output_use_standard_pred

# python scripts/main.py \
#     --config configs/datasets/$REGION.yml  \
#     configs/datasets/${REGION}_ood_test.yml \
#     configs/networks/resnet50.yml \
#     configs/networks/${NETWORK}.yml \
#     configs/pipelines/train/baseline.yml \
#     configs/pipelines/train/train_$METHOD.yml \
#     configs/pipelines/test/test_ood.yml \
#     configs/preprocessors/base_preprocessor.yml \
#     configs/postprocessors/$POSTHOC_METHOD.yml \
#     --use_standard_pred False \
#     --trainer.name $METHOD \
#     --dataset.name $REGION \
#     --dataset.num_classes $NUM_CLASSES \
#     --dataset.train.dataset_class $DATASET_CLASS \
#     --network.pretrained True \
#     --network.checkpoint $WEIGHT_DIR/${METHOD}_${REGION}.pth \
#     --network.backbone.name resnet50 \
#     --num_gpus 1 --num_workers 8 \
#     --dataset.train.batch_size 512 \
#     --merge_option merge \
#     --seed 0 \
#     --output_dir $SCRATCH/open_insect_test/output_use_custom_pred