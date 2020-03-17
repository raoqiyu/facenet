
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import pickle

image_fname     = "./40_Gymnastics_Gymnastics_40_952.jpg"
input_size      = 320
graph           = tf.Graph()


# Load TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path="./model/tflite/facenet.tflite")
interpreter = tf.lite.Interpreter(model_path="./facenet_mobilenet_lf.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#print(input_details)
#print(output_details)

# Test model on random input data.
input_shape = input_details[0]['shape']

image_dir = '../data/lfw/lfw_mtcnnpy_160/'
test_name_fname = './data/pairs.txt'
output_dir = './data/lfw_embeddings/'
  
test_image_names = []
with open(test_name_fname, 'r') as f:
    for line in f.readlines()[1:]:
        name = line.strip().split()[0]
        test_image_names.append(name)

image_embeddings = []
image_names = []

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
        img_ori = np.array(img)
        input_data = np.array(img.resize((224,224))).astype(np.uint8).reshape((1,224,224,3))

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
    
        detection_encodings   = interpreter.get_tensor(output_details[0]['index'])
    
        image_embeddings.append(detection_encodings[0])
        image_names.append(image_fname)


with open('./data/lfw_test_image_embeddigns_mobilenet_lf.pkl','wb') as f:
    pickle.dump(image_embeddings,f)
    pickle.dump(image_names,f)
