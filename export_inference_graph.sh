#!/bin/bash

python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path training_faces_v3/ssd_mobilenet_v2_quantized_300x300_coco.config \
    --trained_checkpoint_prefix training_faces_v3/model.ckpt-253 --output_directory trained-inference/output_inference_graph_v3_2_faces