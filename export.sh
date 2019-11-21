#!/bin/bash

# For person
#python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v2_quantized_300x300_coco.config --trained_checkpoint_prefix training/model.ckpt-572 --output_directory trained-inference/output_inference_graph_v1

# For face
python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path training_faces/ssd_mobilenet_v2_quantized_300x300_coco.config --trained_checkpoint_prefix training_faces/model.ckpt-200 --output_directory trained-inference/output_inference_graph_v1_faces
