# quant

import sys
sys.path.append('/tmp/model/')

import tensorflow as tf
from nets.mobilenet import mobilenet_v2
from tensorflow.python.framework import graph_util


ckpt_file = "./model/20191106-094009/model-20191106-094009.ckpt-250"

output_file = './facenet_mobilenet_lf.pb'

with tf.Graph().as_default():
    with tf.Session() as sess:
        input_data = tf.placeholder(dtype=tf.float32, shape=[None, 224,224,3], name='input')

        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
            logits, end_points = mobilenet_v2.mobilenet(input_data, num_classes=10575)
            prelogits = tf.squeeze(end_points['global_pool'], [1, 2])

        embeddings = tf.identity(prelogits, 'embeddings')


        output_node_names = ['input','embeddings']

        loader = tf.train.Saver()

        # sess.run(tf.global_variables_initializer())

        loader.restore(sess, ckpt_file)

        builder = tf.saved_model.builder.SavedModelBuilder('./model/saved_model/')

        tensor_info_inputs = {
            'inputs': tf.saved_model.utils.build_tensor_info(input_data)}
        tensor_info_outputs = {'embedding':tf.saved_model.utils.build_tensor_info(embeddings)}

        recognition_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=tensor_info_inputs,
                outputs=tensor_info_outputs,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            ))

        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants
                    .DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    recognition_signature,
            },
        )
        builder.save()