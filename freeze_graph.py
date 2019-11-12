# quant

import sys
sys.path.append('/tmp/model/')

import tensorflow as tf
from nets.mobilenet import mobilenet_v2
from tensorflow.python.framework import graph_util


ckpt_file = "./model/20191106-094009/model-20191106-094009.ckpt-101"

output_file = './facenet_mobilenet_lf.pb'

with tf.Graph().as_default():
    with tf.Session() as sess:
        input_data = tf.placeholder(dtype=tf.float32, shape=[None, 224,224,3], name='input')

        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
            logits, end_points = mobilenet_v2.mobilenet(input_data, num_classes=10575)
            prelogits = tf.squeeze(end_points['global_pool'], [1, 2])

        embeddings = tf.identity(prelogits, 'embeddings')

        g = tf.get_default_graph()
        tf.contrib.quantize.create_eval_graph(input_graph=g)

        output_node_names = ['input','embeddings']

        loader = tf.train.Saver()

        # sess.run(tf.global_variables_initializer())

        loader.restore(sess, ckpt_file)

        output_graph_def = graph_util.convert_variables_to_constants(
                    sess,  sess.graph.as_graph_def(), output_node_names,
                )

        with tf.gfile.GFile(output_file, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph: %s" % (len(output_graph_def.node), output_file))


"""
toco --output_file facenet_mobilenet_lf.tflite \
--graph_def_file facenet_mobilenet_lf.pb \
--input_arrays 'input' --input_shapes  '1,224,224,3' \
--output_arrays 'embeddings' --output_format TFLITE \
--mean_values 128 --std_dev_values 64.15 \
--inference_type QUANTIZED_UINT8       

"""

# freeze_graph --input_graph=./yolov3_quant_graph_sh_relu.def --input_checkpoint=checkpoint/yolov3_quant_graph_sh_relu_checkpoint --output_graph=yolov3_frozen_eval_graph_sh_relu.pb --output_node_names=input/input_data,conv_lbbox/BiasAdd,conv_mbbox/BiasAdd,conv_sbbox/BiasAdd


# freeze_graph --input_graph=./yolov3_quant_graph_lf.def --input_checkpoint=checkpoint/yolov3_test_loss=12.3085.ckpt-2 --output_graph=yolov3_frozen_eval_graph.pb --output_node_names=define_input/input_data,define_loss/pred_sbbox/concat,define_loss/pred_mbbox/concat,define_loss/pred_lbbox/concat
