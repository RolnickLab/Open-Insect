#!/bin/bash


bash scripts/eval.sh c-america basics msp resnet50 $SCRATCH/open_insect_camera_ready/weights
bash scripts/eval.sh c-america conf_branch conf_branch conf_branch $SCRATCH/open_insect_camera_ready/weights
bash scripts/eval.sh c-america logitnorm msp resnet50 $SCRATCH/open_insect_camera_ready/weights
bash scripts/eval.sh c-america godin godin godin_net $SCRATCH/open_insect_camera_ready/weights # need to fix
bash scripts/eval.sh c-america rotpred rotpred rot_net $SCRATCH/open_insect_camera_ready/weights
bash scripts/eval.sh c-america oe msp resnet50 $SCRATCH/open_insect_camera_ready/weights
bash scripts/eval.sh c-america udg msp udg_net $SCRATCH/open_insect_camera_ready/weights
bash scripts/eval.sh c-america mixoe msp resnet50 $SCRATCH/open_insect_camera_ready/weights
bash scripts/eval.sh c-america energy msp resnet50 $SCRATCH/open_insect_camera_ready/weights
bash scripts/eval.sh c-america extended msp extended_net $SCRATCH/open_insect_camera_ready/weights
bash scripts/eval.sh c-america novel_branch msp extended_net $SCRATCH/open_insect_camera_ready/weights
