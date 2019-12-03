#!/bin/bash

git clone https://github.com/tensorflow/models.git
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI; make; cp -r pycocotools "../models/research/"

apt-get install protobuf-compiler python-pil python-lxml python-tk
pip install Cython

cd /content/gdrive/My Drive/Desktop/models/research/
protoc object_detection/protos/*.proto --python_out=.

import os
os.environ['PYTHONPATH'] += ':/content/gdrive/My Drive/Desktop/models/research/:/content/gdrive/My Drive/Desktop/models/research/slim'
#'''the lines 2,3 and 4 here are a single line just that the code block wasn't sufficient to occupy it'''

#Always run this every restart of session in GOOGLE COLAB

python setup.py build
python setup.py install


# Remaining time 

import time, psutil
Start = time.time()- psutil.boot_time()
Left= 12*3600 - Start
print('Time remaining for this session is: ', Left/3600)

#rember the last CD you did in order to specify the directory.

!python object_detection/builders/model_builder_test.py