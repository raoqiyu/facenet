import tensorflow as tf
from tensorflow.python.framework import graph_util


detection_graph_path = './model/tflite_graph.pb'
recognition_graph_path = './facenet_mobilenet_lf.pb'
output_file = './model/combine_graph_mobilenet.pb'

detection_graph = tf.GraphDef()
with open(detection_graph_path, 'rb') as f:
    detection_graph.ParseFromString(f.read())

# for node in detection_graph.node:
#     print(node.name)

recognition_graph = tf.GraphDef()
with open(recognition_graph_path, 'rb') as f:
    recognition_graph.ParseFromString(f.read())

with tf.Graph().as_default() as g_combined:
  with tf.Session(graph=g_combined) as sess:

    image_detection = tf.placeholder(tf.float32, shape=[1,320,320,3], name="image_input")
    # image_detection = tf.placeholder(tf.uint8, shape=[None,None,3], name="image_input")
    # width = tf.placeholder(tf.uint8,shape=[1])
    # height = tf.placeholder(tf.uint8,shape=[1])
    # image_size =  tf.concat([[width],[height],[width],[height]],axis=1)
    # image_size =  tf.constant([160,160,160,160])
    # image = tf.reshape(image_detection,[320,320,3])

    # image_detection = tf.reshape(tf.image.resize(image, [320,320], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),[1,320,320,3])
    #
    detection_boxes,detection_classes, detection_scores,num_detections  = tf.import_graph_def(detection_graph,
                                input_map={"image_tensor:0": image_detection},name='detection',
                             return_elements=['detection_boxes:0', 'detection_classes:0',
                                              'detection_scores:0', 'num_detections:0'])
    print(detection_boxes)
    detection_boxes = tf.identity(detection_boxes,'detection_boxes')
    detection_classes = tf.identity(detection_classes,'detection_classes')
    detection_scores = tf.identity(detection_scores, 'detection_scores')
    num_detections = tf.identity(num_detections, 'num_detections')

    # detection_boxes = tf.reshape(detection_boxes, [100,4])
    #Tensor("Reshape_1:0", shape=(100, 4), dtype=float32)
    print(detection_boxes)

    # detection_boxes = tf.cast(detection_boxes*[[320.,320.,320.,320.]],tf.int32)

    detection_boxes_raw = tf.identity(detection_boxes,'detection_boxes_raw')

    #Tensor("detection_boxes_raw:0", shape=(100, 4), dtype=int32)
    print(detection_boxes_raw)

    # detection_image_cropped = []
    # for i in range(1):
    #     # x = detection_boxes_raw[i]
    #     # image_cropped = tf.image.resize_bilinear(image_detection[:,x[0]:x[2]+1,x[1]:x[3]+1,:],[224,224])
    #     image_cropped = image_detection[:,:224,:224,:]
    #     # detection_image_cropped.append(tf.squeeze(image_cropped))

    detection_image_cropped = image_detection[:,:224,:224,:]
    print(detection_image_cropped[0])
    # detection_image_cropped = tf.map_fn(lambda x: tf.image.resize_bilinear(image_detection[:,x[0]:x[2]+1,x[1]:x[3]+1,:],[224,224]),
    #                                     detection_boxes_raw,
    #                                     dtype=tf.float32)

    # detection_image_cropped = tf.stack(detection_image_cropped,axis=0)
    image_recognition = tf.identity(detection_image_cropped,'image_recognition')
    print(image_recognition)

    # detection_boxes_features = tf.import_graph_def(recognition_graph, input_map={"input:0": image_recognition},
    #                          return_elements=['embeddings:0'],name='recognition')
    #
    # detection_boxes_features = tf.identity(detection_boxes_features, 'detection_features')
    #
    # g = tf.get_default_graph()
    # tf.contrib.quantize.create_eval_graph(input_graph=g)

    # freeze combined graph

    summary_writer = tf.summary.FileWriter("./tfborad", sess.graph)

    g_combined_def = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                               ["image_input", 'detection_boxes','detection_classes',
                                                                'detection_scores','num_detections',
                                                                'detection_boxes_raw'])#,'image_recognition',
                                                                # 'detection_features'])


    with tf.gfile.GFile(output_file, 'wb') as f:
        f.write(g_combined_def.SerializeToString())
        print("%d ops in the final graph: %s" % (len(g_combined_def.node), output_file))

"""
toco --output_file model/combine_graph_mobilenet.tflite \
--graph_def_file model/combine_graph_mobilenet.pb \
--input_arrays 'image_input' --input_shapes  '1,320,320,3' \
--output_arrays 'detection_boxes','detection_classes','detection_scores','num_detections','detection_boxes_raw','image_recognition','detection_features' \
--output_format TFLITE \
--mean_values 128 --std_dev_values 128 \
--inference_type QUANTIZED_UINT8 \
--allow_custom_ops \
--default_ranges_min 0 --default_ranges_max 6 \
"""