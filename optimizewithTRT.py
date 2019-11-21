import ctypes
import uff
import tensorrt as trt
import graphsurgeon as gs
import pycuda.driver as cuda
import pycuda.autoinit

ctypes.CDLL("build/libflattenconcat.so")

# Preprocess function to convert TF model to UFF
def ssd_mobilenet_v2_unsupported_nodes_to_plugin_nodes(ssd_graph, input_shape):
    """Makes ssd_graph TensorRT comparible using graphsurgeon.

    This function takes ssd_graph, which contains graphsurgeon
    DynamicGraph data structure. This structure describes frozen Tensorflow
    graph, that can be modified using graphsurgeon (by deleting, adding,
    replacing certain nodes). The graph is modified by removing
    Tensorflow operations that are not supported by TensorRT's UffParser
    and replacing them with custom layer plugin nodes.

    Note: This specific implementation works only for
    ssd_mobilenet_v2_coco_2018_03_29 network.

    Args:
        ssd_graph (gs.DynamicGraph): graph to convert
        input_shape: input shape in CHW format
    Returns:
        gs.DynamicGraph: UffParser compatible SSD graph
    """

    channels, height, width = input_shape

    Input = gs.create_plugin_node(name="Input",
        op="Placeholder",
        dtype=tf.float32,
        shape=[1, channels, height, width])
    PriorBox = gs.create_plugin_node(name="GridAnchor", op="GridAnchor_TRT",
        minSize=0.2,
        maxSize=0.95,
        aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
        variance=[0.1,0.1,0.2,0.2],
        featureMapShapes=[19, 10, 5, 3, 2, 1],
        numLayers=6
    )
    NMS = gs.create_plugin_node(
        name="NMS",
        op="NMS_TRT",
        shareLocation=1,
        varianceEncodedInTarget=0,
        backgroundLabelId=0,
        confidenceThreshold=1e-8,
        nmsThreshold=0.6,
        topK=100,
        keepTopK=100,
        numClasses=91,
        inputOrder=[1, 0, 2],
        confSigmoid=1,
        isNormalized=1
    )
    concat_priorbox = gs.create_node(
        "concat_priorbox",
        op="ConcatV2",
        dtype=tf.float32,
        axis=2
    )
    concat_box_loc = gs.create_plugin_node(
        "concat_box_loc",
        op="FlattenConcat_TRT",
        dtype=tf.float32,
        axis=1,
        ignoreBatch=0
    )
    concat_box_conf = gs.create_plugin_node(
        "concat_box_conf",
        op="FlattenConcat_TRT",
        dtype=tf.float32,
        axis=1,
        ignoreBatch=0
    )

    # Create a mapping of namespace names -> plugin nodes.
    namespace_plugin_map = {
        "MultipleGridAnchorGenerator": PriorBox,
        "Postprocessor": NMS,
        "Preprocessor/map": Input,
        "ToFloat": Input,
        # "image_tensor": Input,
        "Concatenate": concat_priorbox,
        "concat": concat_box_loc,
        "concat_1": concat_box_conf
    }
    for node in ssd_graph.graph_inputs:
        namespace_plugin_map[node.name] = Input

    # Create a new graph by collapsing namespaces
    ssd_graph.collapse_namespaces(namespace_plugin_map)
    # Remove the outputs, so we just have a single output node (NMS).
    # If remove_exclusive_dependencies is True, the whole graph will be removed!
    ssd_graph.remove(ssd_graph.graph_outputs, remove_exclusive_dependencies=False)
    # Disconnect the Input node from NMS, as it expects to have only 3 inputs.
    ssd_graph.find_nodes_by_op("NMS_TRT")[0].input.remove("Input")
    
    return ssd_graph

  
# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
      
      
def allocate_buffers(engine):
    """Allocates host and device buffer for TRT engine inference.

    This function is similair to the one in ../../common.py, but
    converts network outputs (which are np.float32) appropriately
    before writing them to Python buffer. This is needed, since
    TensorRT plugins doesn't support output type description, and
    in our particular case, we use NMS plugin as network output.

    Args:
        engine (trt.ICudaEngine): TensorRT engine

    Returns:
        inputs [HostDeviceMem]: engine input memory
        outputs [HostDeviceMem]: engine output memory
        bindings [int]: buffer to device bindings
        stream (cuda.Stream): cuda stream for engine inference synchronization
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    # Current NMS implementation in TRT only supports DataType.FLOAT but
    # it may change in the future, which could brake this sample here
    # when using lower precision [e.g. NMS output would not be np.float32
    # anymore, even though this is assumed in binding_to_type]
    binding_to_type = {"Input": np.float32, "NMS": np.float32, "NMS_1": np.int32}

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = binding_to_type[str(binding)]
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream
  
# Export UFF model file
ssd_mobilenet_v2_pb_path = "/home/samir/developer/python/learn/models/research/object_detection/samirpc/trained-inference/output_inference_graph_v1_faces/frozen_inference_graph.pb"
output_uff_filename = "/home/samir/developer/python/learn/models/research/object_detection/samirpc/trained-inference/output_inference_graph_v1_faces/frozen_inference_graph.uff"
input_shape = (3, 300, 300)

dynamic_graph = gs.DynamicGraph(ssd_mobilenet_v2_pb_path)
dynamic_graph = ssd_mobilenet_v2_unsupported_nodes_to_plugin_nodes(dynamic_graph, input_shape)

uff.from_tensorflow(dynamic_graph.as_graph_def(), output_nodes=["NMS"], output_filename=output_uff_filename)