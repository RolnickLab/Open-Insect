#!/bin/bash

python download.py --download_dir . --resize_size 224 --region_name "c-america"
python download.py --download_dir . --resize_size 224 --region_name "ne-america"
python download.py --download_dir . --resize_size 224 --region_name "w-europe"

python download_pretrained_weights.py --weight_dir $SCRATCH/download_weights_test 