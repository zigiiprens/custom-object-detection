#!/bin/bash

#python3 cameraTFTRT.py --webcam --model ssd_mobilenet_v2_quantized_300x300_coco --num-classes 1 --build

#python3 cameraTFTRT.py --webcam --model ssd_mobilenet_v2_quantized_300x300_coco --confidence 0.3 --labelmap training_faces_v3/facelabelmap.pbtxt --num-classes 2 --build

python3 cameraTFTRT.py --webcam --confidence 0.3 --model ssd_mobilenet_v2_quantized_300x300_coco --labelmap training_faces_v2/facelabelmap.pbtxt --num-classes 2
