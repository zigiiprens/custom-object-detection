#!/bin/bash

# For person
#python3 train.py --clone_on_cpu=False --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v2_quantized_300x300_coco.config

# For face
python3 train.py --worker_replicas=1 --num_clones=1 --ps_tasks=1 --clone_on_cpu=False --train_dir=training_faces_v2/ --pipeline_config_path=training_faces_v2/ssd_mobilenet_v2_quantized_300x300_coco.config
