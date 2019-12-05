######## Object Detection for Image (PERSON & FACES) #########
#
# Author: Samir
# Date: 14/11/2019

# Some parts of the code is copied from Tensorflow object detection
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb


# Import libraries
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from PIL import Image
from matplotlib import pyplot as plt
from io import StringIO
from collections import defaultdict
import argparse
import cv2
import zipfile
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
print("[INFO] TENSORFLOW VERSION " + tf.__version__)


# Define the video stream
#cap = cv2.VideoCapture("rtsp://10.42.0.202:554/MainStream")
cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams

# What model
directPath = os.getcwd()
print(directPath)

parser = argparse.ArgumentParser()
parser.add_argument("model_option")
args = parser.parse_args()
print(args.model_option)

if str(args.model_option) == "faces":
    MODEL_NAME = os.path.join(
        directPath, 'trained-inference/output_inference_graph_v1_faces')

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join(
        directPath, 'training_faces/facelabelmap.pbtxt')

    # Number of classes to detect
    NUM_CLASSES = 1

elif str(args.model_option) == "person":
    MODEL_NAME = os.path.join(
        directPath, 'trained-inference/output_inference_graph_v1')

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join(directPath, 'training/labelmap.pbtxt')

    # Number of classes to detect
    NUM_CLASSES = 1

elif str(args.model_option) == "mult":
    MODEL_NAME = os.path.join(
        directPath, 'trained-inference/output_inference_graph_v4_faces')

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join(directPath, 'training_faces_v1/facelabelmap.pbtxt')

    # Number of classes to detect
    NUM_CLASSES = 3


elif str(args.model_option) == "mult3":
    MODEL_NAME = os.path.join(
        directPath, 'trained-inference/output_inference_graph_v3_3_faces')

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join(directPath, 'training_faces_v2/facelabelmap.pbtxt')

    # Number of classes to detect
    NUM_CLASSES = 3



# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


frameCount = 0

# Detection
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            frameCount += 1
            if frameCount > 1:

                # Read frame from camera
                ret, image_np = cap.read()
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Extract image tensor
                image_tensor = detection_graph.get_tensor_by_name(
                    'image_tensor:0')
                # Extract detection boxes
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Extract detection scores
                scores = detection_graph.get_tensor_by_name(
                    'detection_scores:0')
                # Extract detection classes
                classes = detection_graph.get_tensor_by_name(
                    'detection_classes:0')
                # Extract number of detectionsd
                num_detections = detection_graph.get_tensor_by_name(
                    'num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    min_score_thresh=0.70,
                    line_thickness=2)
                # Visualization of coordinates of the results of a detection.
                # This is a tuned version of the original
                # "visualize_boxes_and_labels_on_image_array" function under <visualization_utils.py>
                coordinates = vis_util.return_coordinates(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    min_score_thresh=0.70,
                    line_thickness=2)
                
                print("\n")
                print("[INFO COORDINATES] => " + str(coordinates))
                print("\n")

                # Display output
                cv2.imshow('object detection',
                           cv2.resize(image_np, (1200, 900)))

                key = cv2.waitKey(1)
                if key == 27:  # ESC key: quit program
                    cv2.destroyAllWindows()
                    break

                frameCount = 0

            else:
                continue
