# CustomObjectDetection

# Tensorflow object detection - Custom object detector

This is about custom object detector using tensorflow object detection API. In this project trying to detect person/faces. Please refer step by step.

## Preparing resources for tensorflow custom object detection.

1. Download the images of the object what you want to detect.
2. Download the [tensorflow models](https://github.com/tensorflow/models) repo and do installation steps in which is given at [object detection](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

# OR

1. Run this the ```requirement.sh``` file somewhere.

## Labeling image and exporting csv

1. Labeling image is main the process of object detection. It is the process of adding bounding boxes to the objects in the images.

2. In this project [labelImg](https://github.com/tzutalin/labelImg) used for image labeling.

3. After Labeled images the folder structure images have all the images. The subdirectory train and test have the xml files. The folder structure was given below. Create folder named data also in main folder.

	```
	Pre Processing
	|
	└───/images(images_faces)
	│   │
	│   └───/train
	│   │   	// .jpg files of train images 
	│   │		// .xml files of train images
	│   └───/test 
	│		  	// .xml files of test images
	│			// .jpg files of train images
	│   
	└─── xml_to_csv.py
	│   
	└─── generate_tfrecord.py
	```

4. Execute the xml\_to\_csv.py file. It will create the test.csv and train.csv inside the data folder.

## Create tf records

1. Before creating tf record change the lable row\_label to repected class at line 30 in generate_tfrecord.py. Here I have one object so I changed the label as 'person' or 'faces'.

2. After changing line execute (sh generate_tfrecord.sh). It will create the test.record and train.record inside the data folder.

## Create .pbtxt(label map) file

The label map file will contain the labels and ids of the object. In this case the object was only one. So the pbtxt file have label for only one object.

## Create .config file

1. The creation of .config file for explained in this [link](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md).

2. To complete this process the .ckpt file also needed. In this case .ckpt inherited from [ssd\_mobilenet\_v2\_coco\](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz). Download the model and move it to object\_detection folder


## Final folder structure

1. Open tensorflow models directory and execute following commands.

	```
	# From tensorflow/models/research/
	protoc object_detection/protos/*.proto --python_out=.
	
	# From tensorflow/models/research/
	export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim 
	
	```

2. Now move the .pbtxt, .records and .csv files to /research/object_detection/data

3. split images as train and evol folder and move it to /research/object_detection/models/model folder. Also move .config file also to this folder.

4. The final folder structure as follows.

	```
	CustomObjectDetection
	|
	└───/images
	│   │   
	│   └─── test.record 
	│   └─── train.record 
	│   └─── test_labels.csv
	│   └─── train_labels.csv
	└───/images_faces
	│   │   
	│   └─── test.record 
	│   └─── train.record 
	│   └─── test_labels.csv
	│   └─── train_labels.csv 
	|
	└───/trained_inference
	│	│
	│	└───/output_inference_graph_v1
	│	│	│
	│	│	└───/saved_model
	│	│	│	// saved final model
	│	│	└─── Other files (frozen_inference...)
	│	│
	│	└───/output_inference_graph_v1_faces
	│		│
	│		└───/saved_model
	│		│	// saved final model
	│		└─── Other files (frozen_inference...)
	│
	└───/training
	│   │
	│   └─── labelmap.pbtxt
	│   └─── Other files
	│
	└───/training_faces
	│   │
	│   └─── labelmap.pbtxt
	│   └─── Other files
	│
	└───/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03
	│	    │
	│	    └─── model.ckpt
	│	    │
	│	    └─── Other files
	│   
	└─── Other folders and files
	```


## Trainning a model

Execute following command in terminal.

```
sh train.sh
```

Above command will train model locally. If want to train in cloud please refer [docs](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_cloud.md)

The trainned model will be available at ```trained_inference/output_inference_graph_v1/saved_model``` folder.

## Export Frozen Graph

To export "trained_inference/output_inference_graph_v1/frozen_inference_graph.pb" file execute following commad.

```
sh export.sh
```
	
## Run and test model

1. Run ```python3 webcam.py``` to work with generated TF model
2. Run ```sh cameraTFTRT.sh``` to work with optimized TRT model

	

