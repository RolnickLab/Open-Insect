#!/bin/bash
#SBATCH --job-name=opengan
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=long             
#SBATCH --cpus-per-task=8    
#SBATCH --gres=gpu:rtx8000:1   

REGION=$1
METHOD=$2
POSTHOC_METHOD=$3
NETWORK=$4
WEIGHT_DIR=$5

module load anaconda/3

conda activate oi_env

# evaluation
python scripts/main.py \
    --config configs/datasets/$REGION.yml  \
    configs/datasets/${REGION}_ood_test.yml \
    configs/networks/opengan.yml \
    configs/pipelines/test/test_opengan.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/opengan.yml \
    --use_standard_pred False \
    --dataset.name $REGION \
    --num_workers 8 \
    --network.nc 2048 \
    --network.weight_dir $WEIGHT_DIR \
    --network.backbone.name resnet50 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint $WEIGHT_DIR/basics_${REGION}.pth \
    --output_dir $SCRATCH/open_insect_test/output \
    --merge_option merge \
    --seed  0
