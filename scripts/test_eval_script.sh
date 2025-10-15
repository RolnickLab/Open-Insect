#!/bin/bash
module load anaconda/3

conda activate oi_env


bash scripts/eval_arpl.sh c-america arpl msp arpl_net $SCRATCH/open_insect_camera_ready/weights


bash scripts/eval.sh c-america basics temperature_scaling resnet50 $SCRATCH/open_insect_camera_ready/weights

# bash scripts/eval.sh c-america basics msp resnet50 $SCRATCH/open_insect_camera_ready/weights
# --- DONE ---
# bash scripts/train.sh c-america basics resnet50
# bash scripts/train.sh c-america conf_branch conf_branch
# bash scripts/train.sh c-america logitnorm resnet50
# bash scripts/train.sh c-america rotpred rot_net 
# bash scripts/train.sh c-america oe resnet50
# bash scripts/train.sh c-america energy resnet50


# --- DEBUGGING ---
# bash scripts/train.sh c-america godin godin_net # - can't load backbone
# bash scripts/train.sh c-america udg udg_net # clustering failed
# bash scripts/train.sh c-america mixoe resnet50

# --- TO DO ---


# bash scripts/train.sh c-america extended extended_net
# bash scripts/train.sh c-america novel_branch extended_net

# neco
# for posthoc_method in   rp_msp rp_odin rp_ebo rp_gradnorm; do
#   echo "$posthoc_method"
#   bash scripts/eval.sh c-america basics $posthoc_method resnet50 $SCRATCH/open_insect_camera_ready/weights

# done



# for posthoc_method in openmax msp temperature_scaling odin mds mds_ensemble gram ebo gradnorm react mls klm vim knn dice; do
#   echo "$posthoc_method"
#   bash scripts/eval.sh c-america basics $posthoc_method resnet50 $SCRATCH/open_insect_camera_ready/weights

# done


# for posthoc_method in gradnorm rankfeat ash she neco fdbd rp_msp rp_odin rp_ebo rp_gradnorm nci; do
#   echo "$posthoc_method"
#   bash scripts/eval.sh c-america basics $posthoc_method resnet50 $SCRATCH/open_insect_camera_ready/weights

# done

# for posthoc_method in neco rp_msp rp_odin rp_ebo rp_gradnorm nci; do
#   echo "$posthoc_method"
#   bash scripts/eval.sh c-america basics $posthoc_method resnet50 $SCRATCH/open_insect_camera_ready/weights

# done


# for posthoc_method in nci; do
#   echo "$posthoc_method"
#   bash scripts/eval.sh c-america basics $posthoc_method resnet50 $SCRATCH/open_insect_camera_ready/weights

# done



# bash scripts/eval.sh c-america conf_branch conf_branch conf_branch $SCRATCH/open_insect_camera_ready/weights
# bash scripts/eval.sh c-america logitnorm msp resnet50 $SCRATCH/open_insect_camera_ready/weights
# bash scripts/eval.sh c-america godin godin godin_net $SCRATCH/open_insect_camera_ready/weights 
# bash scripts/eval.sh c-america rotpred rotpred rot_net $SCRATCH/open_insect_camera_ready/weights
# bash scripts/eval.sh c-america oe msp resnet50 $SCRATCH/open_insect_camera_ready/weights
# bash scripts/eval.sh c-america udg msp udg_net $SCRATCH/open_insect_camera_ready/weights
# bash scripts/eval.sh c-america mixoe msp resnet50 $SCRATCH/open_insect_camera_ready/weights
# bash scripts/eval.sh c-america energy msp resnet50 $SCRATCH/open_insect_camera_ready/weights
# bash scripts/eval.sh c-america extended msp extended_net $SCRATCH/open_insect_camera_ready/weights
# bash scripts/eval.sh c-america novel_branch msp extended_net $SCRATCH/open_insect_camera_ready/weights