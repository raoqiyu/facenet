import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import pickle
from src import facenet

from sklearn.metrics.pairwise import cosine_distances,cosine_similarity,euclidean_distances


import importlib

import tensorflow as tf

network = importlib.import_module('src.models.inception_resnet_v1')

model_dir = "./model/20180402-114759/20180402-114759.pb"
ckpt_file = "./model/20180402-114759/model-20180402-114759.ckpt-275"

def exporter():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # input_data = tf.placeholder(dtype=tf.float32, shape=[None, 224,224,3], name='input')
            #
            # output, _ = network.inference(input_data, keep_probability=1, phase_train=False, bottleneck_layer_size=512)
            #
            # embeddings = tf.identity(output, 'embeddings')
            #
            # output_node_names = ['input','embeddings']
            #
            # loader = tf.train.Saver()
            #
            # # sess.run(tf.global_variables_initializer())
            #
            # loader.restore(sess, ckpt_file)

            facenet.load_model(model_dir)

            # Get input and output tensors
            input_data = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")



            tensor_info_inputs = {
                'inputs': tf.saved_model.utils.build_tensor_info(input_data),
                'phase_train': tf.saved_model.utils.build_tensor_info(phase_train_placeholder)}
            tensor_info_outputs = {'embedding':tf.saved_model.utils.build_tensor_info(embeddings)}

            recognition_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs=tensor_info_inputs,
                    outputs=tensor_info_outputs,
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                ))

            builder = tf.saved_model.builder.SavedModelBuilder('./model/saved_model/1')
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


image_dir = '../data/lfw/lfw_mtcnnpy_160/'
test_name_fname = './data/pairs.txt'
output_dir = './data/lfw_embeddings/'

test_image_names = []
with open(test_name_fname, 'r') as f:
    for line in f.readlines()[1:]:
        name = line.strip().split()[0]
        test_image_names.append(name)
batch_size = 32

def get_batch_data():
    batch_data,batch_fname = [], []
    for image_name in tqdm(test_image_names):
        image_path = os.path.join(image_dir, image_name)
        i = 0
        for image_fname in os.listdir(image_path):
            if i > 3:
                break
            i += 1
            if not image_fname.endswith('.png'):
                continue
            img = Image.open(os.path.join(image_path, image_fname))
            input_data = np.array(img.resize((160, 160))).astype(np.float32)
            input_data = facenet.prewhiten(input_data)

            batch_data.append(input_data)
            batch_fname.append(image_fname)

            if len(batch_data) == batch_size:
                yield  batch_data,batch_fname
                batch_data, batch_fname = [], []

    if len(batch_data) > 0 :
        yield batch_data,batch_fname

def get_batch_data_test():
    image_path_list = ['./data/compare/Abel_Pacheco_0003.png','./data/compare/Abel_Pacheco_0004.png']
    batch_data, batch_fname = [], []
    for image_path in image_path_list:
        img = Image.open(image_path)
        input_data = np.array(img.resize((160, 160))).astype(np.float32)
        input_data = facenet.prewhiten(input_data)
        Image.fromarray(input_data.astype(np.uint8)).show()
        batch_data.append(input_data)
        batch_fname.append(image_path)
    return batch_data, batch_fname


def test():
    image_embeddings, image_fnames = [], []
    with tf.Session() as sess:
        # input_data = tf.placeholder(dtype=tf.float32, shape=[None, 160, 160, 3], name='input')
        # phase_train_placeholder =  tf.placeholder(dtype=tf.bool, name='phase_train')
        # embeddings, _ = network.inference(input_data, keep_probability=1, phase_train=phase_train_placeholder, bottleneck_layer_size=512)
        # output_node_names = ['input', 'embeddings']
        # loader = tf.train.Saver()
        # sess.run(tf.global_variables_initializer())
        # loader.restore(sess, ckpt_file)
        #
        # Load the model
        facenet.load_model(model_dir)

        # Get input and output tensors
        input_data = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        batch_data, batch_fname = get_batch_data_test()

        # for batch_data,batch_fname in tqdm(get_batch_data()):
        batch_embedding = sess.run(embeddings, feed_dict={input_data:batch_data,phase_train_placeholder:False})
        image_embeddings.append(batch_embedding)
        image_fnames.extend(batch_fname)

        print(euclidean_distances(batch_embedding,batch_embedding))
        print(image_fnames)
        print(image_embeddings)
    with open('./data/lfw_test_image_embeddigns_inception_v1.pkl','wb') as f:
        pickle.dump(image_embeddings,f)
        pickle.dump(image_fnames,f)

if __name__ == '__main__':
    test()
    # exporter()