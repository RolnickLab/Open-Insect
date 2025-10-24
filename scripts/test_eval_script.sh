#!/bin/bash
module load anaconda/3

conda activate oi_env


# sbatch scripts/eval.sh w-europe basics msp resnet50 $SCRATCH/open_insect_camera_ready/weights

# sbatch scripts/eval_opengan.sh c-america opengan opengan resnet50 $SCRATCH/open_insect_camera_ready/weights
sbatch scripts/eval_opengan.sh ne-america opengan opengan resnet50 $SCRATCH/open_insect_camera_ready/weights
sbatch scripts/eval_opengan.sh w-europe opengan opengan resnet50 $SCRATCH/open_insect_camera_ready/weights

# sbatch scripts/eval.sh ne-america basics vim resnet50 $SCRATCH/open_insect_camera_ready/weights

# bash scripts/eval.sh c-america basics dice resnet50 $SCRATCH/open_insect_camera_ready/weights


# sbatch scripts/eval.sh ne-america basics neco resnet50 $SCRATCH/open_insect_camera_ready/weights
# sbatch scripts/eval.sh ne-america basics vim resnet50 $SCRATCH/open_insect_camera_ready/weights



## NEED TO REUN THESE ##
# for posthoc_method in  openmax msp temperature_scaling odin mds gram ebo gradnorm mls klm vim knn  gradnorm ash she neco fdbd rp_msp rp_odin rp_ebo rp_gradnorm nci; do
#   sbatch scripts/eval.sh c-america basics $posthoc_method resnet50 $SCRATCH/open_insect_camera_ready/weights
# done

# for posthoc_method in rp_msp rp_odin rp_ebo rp_gradnorm; do
#   sbatch scripts/eval.sh c-america basics $posthoc_method resnet50 $SCRATCH/open_insect_camera_ready/weights
# done

# sbatch scripts/eval.sh c-america basics  resnet50 $SCRATCH/open_insect_camera_ready/weights

# sbatch scripts/eval.sh c-america basics dice resnet50 $SCRATCH/open_insect_camera_ready/weights
# sbatch scripts/eval.sh c-america basics react resnet50 $SCRATCH/open_insect_camera_ready/weights
# sbatch scripts/eval.sh c-america basics rankfeat resnet50 $SCRATCH/open_insect_camera_ready/weights
# sbatch scripts/eval.sh c-america basics mds_ensemble resnet50 $SCRATCH/open_insect_camera_ready/weights


# sbatch scripts/eval.sh w-europe basics dice resnet50 $SCRATCH/open_insect_camera_ready/weights
# sbatch scripts/eval.sh w-europe basics react resnet50 $SCRATCH/open_insect_camera_ready/weights
# sbatch scripts/eval.sh w-europe basics rankfeat resnet50 $SCRATCH/open_insect_camera_ready/weights
# sbatch scripts/eval.sh w-europe basics mds_ensemble resnet50 $SCRATCH/open_insect_camera_ready/weights


# sbatch scripts/eval.sh ne-america basics dice resnet50 $SCRATCH/open_insect_camera_ready/weights
# sbatch scripts/eval.sh ne-america basics react resnet50 $SCRATCH/open_insect_camera_ready/weights
# sbatch scripts/eval.sh ne-america basics rankfeat resnet50 $SCRATCH/open_insect_camera_ready/weights
# sbatch scripts/eval.sh ne-america basics ash resnet50 $SCRATCH/open_insect_camera_ready/weights

# sbatch scripts/eval.sh c-america basics rmds resnet50 $SCRATCH/open_insect_camera_ready/weights


# bash scripts/eval.sh c-america basics rp_msp resnet50 $SCRATCH/open_insect_camera_ready/weights
# bash scripts/eval.sh c-america basics rp_ebo resnet50 $SCRATCH/open_insect_camera_ready/weights
# bash scripts/eval.sh c-america basics rp_odin resnet50 $SCRATCH/open_insect_camera_ready/weights
# bash scripts/eval.sh c-america basics rp_gradnorm resnet50 $SCRATCH/open_insect_camera_ready/weights



# sbatch scripts/eval_arpl.sh c-america arpl msp arpl_net $SCRATCH/open_insect_camera_ready/weights
# sbatch scripts/eval_arpl.sh ne-america arpl msp arpl_net $SCRATCH/open_insect_camera_ready/weights
# sbatch scripts/eval_arpl.sh w-europe arpl msp arpl_net $SCRATCH/open_insect_camera_ready/weights


# bash scripts/eval.sh c-america basics rp_msp resnet50 $SCRATCH/open_insect_camera_ready/weights

# bash scripts/eval.sh c-america basics msp resnet50 $SCRATCH/open_insect_camera_ready/weights
# --- DONE ---
# bash scripts/train.sh c-america basics resnet50 $SCRATCH/open_insect_test
# bash scripts/train.sh c-america conf_branch conf_branch $SCRATCH/open_insect_test
# bash scripts/train.sh c-america logitnorm resnet50 $SCRATCH/open_insect_test
# bash scripts/train.sh c-america rotpred rot_net $SCRATCH/open_insect_test
# bash scripts/train.sh c-america oe resnet50 $SCRATCH/open_insect_test
# bash scripts/train.sh c-america energy resnet50 $SCRATCH/open_insect_test
# bash scripts/train.sh c-america extended extended_net $SCRATCH/open_insect_test
# bash scripts/train.sh c-america novel_branch extended_net $SCRATCH/open_insect_test  $SCRATCH/open_insect_camera_ready/weights/basics_c-america.pth


# --- DEBUGGING ---
# bash scripts/train.sh c-america godin godin_net # - can't load backbone
# bash scripts/train.sh c-america udg udg_net # clustering failed
# bash scripts/train.sh c-america mixoe resnet50

# --- TO DO ---

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