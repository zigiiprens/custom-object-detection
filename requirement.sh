#!/bin/bash

git clone https://github.com/tensorflow/models.git
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI 
make
cp -r cocoapi/PythonAPI/pycocotools models/research
apt-get install protobuf-compiler python-pil python-lxml python-tk
pip install Cython
cd models/research/
protoc object_detection/protos/*.proto --python_out=.

# Hense I isntalled it under /home/username/developer/python/learn/
# You can do ====>
# export PYTHONPATH=$PYTHONPATH:./models/research:./models/research/slim
export PYTHONPATH=$PYTHONPATH:/home/username/developer/python/learn/models/research:/home/username/developer/python/learn/models/research/slim

python3 setup.py build
python3 setup.py install