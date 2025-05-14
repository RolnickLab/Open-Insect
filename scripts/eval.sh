#!/bin/bash

python $CURDIR/scripts/eval.py \
    --config configs/datasets/<dataset config>.yml  \
      configs/datasets/<ood dataset config>.yml \
      configs/preprocessors/base_preprocessor.yml \
      configs/networks/<network config>.yml \
      configs/pipelines/test/test_ood.yml \
      configs/postprocessors/<the OOD detection method>.yml \
    --network.checkpoint <path to the checkpoint>  \
    --trainer.name <the method used to train the model> \
    --dataset.train.batch_size 512 \
    --num_gpus 1 --num_workers 8 \
    --merge_option merge \
    --seed 0 