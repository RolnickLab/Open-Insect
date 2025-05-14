#!/bin/bash

python download.py --download_dir $SCRATCH/hf_test --resize_size 224 --region_name "c-america"
python download.py --download_dir $SCRATCH/hf_test --resize_size 224 --region_name "ne-america"
python download.py --download_dir $SCRATCH/hf_test --resize_size 224 --region_name "w-europe"