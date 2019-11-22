#!/bin/bash

#python3 cameraTFTRT.py --webcam --model ssd_mobilenet_v2_quantized_300x300_coco --num-classes 1 --build

python3 cameraTFTRT.py --rtsp --uri fake --model ssd_mobilenet_v2_quantized_300x300_coco --num-classes 1 --build