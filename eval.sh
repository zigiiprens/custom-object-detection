#!/bin/bash

# For person
#python3 eval.py --logtossderr --pipeline_config_path=training/ssd_mobilenet_v2_quantized_300x300_coco.config --checkpoint_dir=training/ --eval_dir=training/

# For face
python3 eval.py --logtossderr --pipeline_config_path=training_faces_v1/ssd_mobilenet_v2_quantized_300x300_coco.config --checkpoint_dir=training_faces_v1/ --eval_dir=eval_faces_v2/
