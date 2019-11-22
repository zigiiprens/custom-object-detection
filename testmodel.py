from tensorflow.python.client import timeline
from io import BytesIO
from PIL import Image, ImageDraw
import time
import requests
import json
import numpy as np
import sys
import os
import tensorflow as tf
print("[INFO] TENSORFLOW VERSION " + tf.__version__)
print(tf.test.is_gpu_available())

def load_graph_def(model_path) -> tf.GraphDef:
    """
    Helper function to read model protobuf and return graph definition
    Args:
        model_path: Path to frozen model .pb file

    Returns:
        graph_def: graph definition

    """
    with tf.device('/device:XLA_GPU:0'):
        graph_def = tf.GraphDef()
        with open(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())

    return graph_def


def get_iamge_by_url(url: str):
    """ Get image data from url"""
    res = requests.get(url)
    return Image.open(BytesIO(res.content))


def draw_bbox_and_label_in_image(img, boxes, num_detections, box_width=3):
    """
    Draw bounding boxes and class labels in images
    :param img: PIL Image or np arrays
    :param boxes: in size [num_detections, 4], contains xys or boxes
    :param box_width: the width of boxes
    :return: Image
    """

    draw = ImageDraw.Draw(img)
    width, height = img.size

    for i in range(num_detections):
        ymin, xmin, ymax, xmax = boxes[i]

        ymin = int(ymin * height)
        ymax = int(ymax * height)
        xmin = int(xmin * width)
        xmax = int(xmax * width)

        class_color = "LimeGreen"

        draw.line([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin,
                                                              ymax), (xmin, ymin)], width=box_width, fill=class_color)

    return img


def run_ssd_mobilenet_v2_tf(image: Image, frozen_model_path: str, record_trace=False, trace_filename="ssd_mobilenet_v2.json"):
    """ Run the model and report the average inference time, return inference time and output for sanity check """
    input_tensor_name = "image_tensor:0"
    output_tensor_names = [
        'detection_boxes:0', 'detection_classes:0', 'detection_scores:0', 'num_detections:0']
    ssd_mobilenet_v2_graph_def = load_graph_def(frozen_model_path)

    # Preprocess the image
    image_tensor = image.resize((300, 300))
    image_tensor = np.array(image_tensor)
    image_tensor = np.expand_dims(image_tensor, axis=0)

    with tf.Graph().as_default() as g:
        tf.import_graph_def(ssd_mobilenet_v2_graph_def, name='')
        input_tensor = g.get_tensor_by_name(input_tensor_name)
        output_tensors = [g.get_tensor_by_name(
            name) for name in output_tensor_names]

        with tf.Session(graph=g) as sess:
            # The first run will generally take longer, so we feed some random data
            # to warm up the session
            sess.run(output_tensors, feed_dict={
                     input_tensor: np.random.randint(0, 255, (1, 300, 300, 3))})

            # Add metadata to record runtime performance for further analysis
            if record_trace:
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                options = None
                run_metadata = None

            inference_time = []

            if record_trace:
                start = time.time()
                outputs = sess.run(output_tensors, feed_dict={input_tensor: image_tensor},
                                   options=options, run_metadata=run_metadata)
                inference_time.append((time.time()-start)*1000.)  # in ms

                # Write metadata
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()

                directPath = os.getcwd()
                print(directPath)
                with open('trained-inference/output_inference_graph_v1_faces/json/' + trace_filename, 'w+') as f:
                    if f.closed:
                        print("[INFO FILE] Closed")
                    else:
                        print("[INFO FILE] Openned")
                        f.write(chrome_trace)
            else:
                # Run multiple times to get meaningful statistic
                for i in range(30):
                    start = time.time()
                    outputs = sess.run(output_tensors, feed_dict={
                                       input_tensor: image_tensor})
                    inference_time.append((time.time()-start)*1000.)  # in ms
                print("SSD MobileNet V2 model inference time: %.2f ms" %
                      np.mean(inference_time))

    return outputs, inference_time


def run_ssd_mobilenet_v2_tf_optimized_CPU(image: Image, frozen_model_path: str, record_trace=False, trace_filename="ssd_mobilenet_v2_cpu.json"):
    """ Run the model and report the average inference time, return inference time and output for sanity check """
    input_tensor_name = "image_tensor:0"
    output_tensor_names = [
        'detection_boxes:0', 'detection_classes:0', 'detection_scores:0', 'num_detections:0']
    ssd_mobilenet_v2_optimized_graph_def = load_graph_def(frozen_model_path)

    for node in ssd_mobilenet_v2_optimized_graph_def.node:
        if 'NonMaxSuppression' in node.name:
            node.device = '/device:CPU:0'

    # Preprocess the image
    image_tensor = image.resize((300, 300))
    image_tensor = np.array(image_tensor)
    image_tensor = np.expand_dims(image_tensor, axis=0)

    with tf.Graph().as_default() as g:
        tf.import_graph_def(ssd_mobilenet_v2_optimized_graph_def, name='')
        input_tensor = g.get_tensor_by_name(input_tensor_name)
        output_tensors = [g.get_tensor_by_name(
            name) for name in output_tensor_names]

        with tf.Session(graph=g) as sess:
            # The first run will generally take longer, so we feed some random data
            # to warm up the session
            sess.run(output_tensors, feed_dict={
                     input_tensor: np.random.randint(0, 255, (1, 300, 300, 3))})

            # Add metadata to record runtime performance for further analysis
            if record_trace:
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                options = None
                run_metadata = None

            inference_time = []

            if record_trace:
                start = time.time()
                outputs = sess.run(output_tensors, feed_dict={input_tensor: image_tensor},
                                   options=options, run_metadata=run_metadata)
                inference_time.append((time.time()-start)*1000.)  # in ms

                # Write metadata
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()

                directPath = os.getcwd()
                print(directPath)
                with open('trained-inference/output_inference_graph_v1_faces/json/' + trace_filename, 'w+') as f:                    
                    print("[INFO FILE] Openned")
                    f.write(chrome_trace)
            else:
                # Run multiple times to get meaningful statistic
                for i in range(30):
                    start = time.time()
                    outputs = sess.run(output_tensors, feed_dict={
                                       input_tensor: image_tensor})
                    inference_time.append((time.time()-start)*1000.)  # in ms
                print("SSD MobileNet V2 model inference time: %.2f ms" %
                      np.mean(inference_time))

    return outputs, inference_time

# What model
directPath = os.getcwd()
print(directPath)
MODEL_PATH = os.path.join(
        directPath, 'trained-inference/output_inference_graph_v1_faces/frozen_inference_graph.pb')

# Download image
ssd_mobilenet_v2_origin_path = MODEL_PATH
#image = get_iamge_by_url("https://leblogdeflorencia.files.wordpress.com/2011/04/img_9962.jpg")
image = Image.open("images/test/person_014.jpg")

# Run test
outputs, inference_time = run_ssd_mobilenet_v2_tf(
    image, ssd_mobilenet_v2_origin_path,True)
print("[INFO] Normal " + str(inference_time))
outputs, inference_time_optimized_CPU = run_ssd_mobilenet_v2_tf_optimized_CPU(
    image, ssd_mobilenet_v2_origin_path,True)
print("[INFO] With CPU optimized " + str(inference_time_optimized_CPU))

# Save resutl to disk, for simplicity, we use JSON here
tf_result = list(inference_time)
with open("tf_result.json", 'w+') as f:
    f.write(json.dumps({'inference_time':inference_time, 'inference_time_CPU': inference_time_optimized_CPU}, sort_keys=True))

# Sanity check
draw_image = draw_bbox_and_label_in_image(
    image, outputs[0][0], int(outputs[3][0]))

# Display output
draw_image.show()
