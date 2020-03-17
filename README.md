# Face Recognition using Tensorflow [![Build Status][travis-image]][travis]

[travis-image]: http://travis-ci.org/davidsandberg/facenet.svg?branch=master
[travis]: http://travis-ci.org/davidsandberg/facenet

This is a TensorFlow implementation of the face recognizer described in the paper
["FaceNet: A Unified Embedding for Face Recognition and Clustering"](http://arxiv.org/abs/1503.03832). The project also uses ideas from the paper ["Deep Face Recognition"](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf) from the [Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/) at Oxford.

## Compatibility
The code is tested using Tensorflow r1.7 under Ubuntu 14.04 with Python 2.7 and Python 3.5. The test cases can be found [here](https://github.com/davidsandberg/facenet/tree/master/test) and the results can be found [here](http://travis-ci.org/davidsandberg/facenet).

## News
| Date     | Update |
|----------|--------|
| 2018-04-10 | Added new models trained on Casia-WebFace and VGGFace2 (see below). Note that the models uses fixed image standardization (see [wiki](https://github.com/davidsandberg/facenet/wiki/Training-using-the-VGGFace2-dataset)). |
| 2018-03-31 | Added a new, more flexible input pipeline as well as a bunch of minor updates. |
| 2017-05-13 | Removed a bunch of older non-slim models. Moved the last bottleneck layer into the respective models. Corrected normalization of Center Loss. |
| 2017-05-06 | Added code to [train a classifier on your own images](https://github.com/davidsandberg/facenet/wiki/Train-a-classifier-on-own-images). Renamed facenet_train.py to train_tripletloss.py and facenet_train_classifier.py to train_softmax.py. |
| 2017-03-02 | Added pretrained models that generate 128-dimensional embeddings.|
| 2017-02-22 | Updated to Tensorflow r1.0. Added Continuous Integration using Travis-CI.|
| 2017-02-03 | Added models where only trainable variables has been stored in the checkpoint. These are therefore significantly smaller. |
| 2017-01-27 | Added a model trained on a subset of the MS-Celeb-1M dataset. The LFW accuracy of this model is around 0.994. |
| 2017&#8209;01&#8209;02 | Updated to run with Tensorflow r0.12. Not sure if it runs with older versions of Tensorflow though.   |

## Pre-trained models
| Model name      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|-------------|
| [20180408-102900](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) | 0.9905        | CASIA-WebFace    | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |
| [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) | 0.9965        | VGGFace2      | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |

NOTE: If you use any of the models, please do not forget to give proper credit to those providing the training dataset as well.

## Inspiration
The code is heavily inspired by the [OpenFace](https://github.com/cmusatyalab/openface) implementation.

## Training data
The [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) dataset has been used for training. This training set consists of total of 453 453 images over 10 575 identities after face detection. Some performance improvement has been seen if the dataset has been filtered before training. Some more information about how this was done will come later.
The best performing model has been trained on the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset consisting of ~3.3M faces and ~9000 classes.

## Pre-processing

### Face alignment using MTCNN
One problem with the above approach seems to be that the Dlib face detector misses some of the hard examples (partial occlusion, silhouettes, etc). This makes the training set too "easy" which causes the model to perform worse on other benchmarks.
To solve this, other face landmark detectors has been tested. One face landmark detector that has proven to work very well in this setting is the
[Multi-task CNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html). A Matlab/Caffe implementation can be found [here](https://github.com/kpzhang93/MTCNN_face_detection_alignment) and this has been used for face alignment with very good results. A Python/Tensorflow implementation of MTCNN can be found [here](https://github.com/davidsandberg/facenet/tree/master/src/align). This implementation does not give identical results to the Matlab/Caffe implementation but the performance is very similar.

## Running training
Currently, the best results are achieved by training the model using softmax loss. Details on how to train a model using softmax loss on the CASIA-WebFace dataset can be found on the page [Classifier training of Inception-ResNet-v1](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1) and .

## Pre-trained models
### Inception-ResNet-v1 model
A couple of pretrained models are provided. They are trained using softmax loss with the Inception-Resnet-v1 model. The datasets has been aligned using [MTCNN](https://github.com/davidsandberg/facenet/tree/master/src/align).

## Performance
The accuracy on LFW for the model [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) is 0.99650+-0.00252. A description of how to run the test can be found on the page [Validate on LFW](https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw). Note that the input images to the model need to be standardized using fixed image standardization (use the option `--use_fixed_image_standardization` when running e.g. `validate_on_lfw.py`).

## CASIA data
Number of classes in training set: 10575
Number of examples in training set: 462406
Number of classes in validation set: 10575
Number of examples in validation set: 29136


## Train
160 + 22

160 + 64 
python src/align/align_dataset_mtcnn.py ./dataset/CASIA-WebFace ./dataset/casia_mtcnnpy_182/ --image_size 182 --margin 44

python src/train_softmax.py \
--logs_base_dir ./log/facenet/ \
--models_base_dir ./model/ \
--pretrained_model ./model/20191029-134852/model-20191029-134852.ckpt-90 \
--data_dir ./dataset/casia_mtcnnpy_182/ \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--lfw_dir /home/rqy/Project/data/lfw_mtcnnpy_160/ \
--optimizer ADAM \
--learning_rate -1 \
--max_nrof_epochs 150 \
--keep_probability 0.8 \
--random_crop \
--random_flip \
--use_fixed_image_standardization \
--learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
--weight_decay 5e-4 \
--embedding_size 512 \
--lfw_distance_metric 1 \
--lfw_use_flipped_images \
--lfw_subtract_mean \
--validation_set_split_ratio 0.05 \
--validate_every_n_epochs 5 \
--prelogits_norm_loss_factor 5e-4


sh
python src/train_softmax_mobilenet.py \
--logs_base_dir ./log/facenet/ \
--models_base_dir ./model/ \
--data_dir ./dataset/casia_mtcnnpy_182/ \
--batch_size 90 \
--image_size 160 \
--lfw_dir /home/rqy/Project/data/lfw_mtcnnpy_160/ \
--pretrained_model ./model/20191101-100725/model-20191101-100725.ckpt-150 \
--optimizer ADAM \
--learning_rate -1 \
--max_nrof_epochs 150 \
--keep_probability 0.8 \
--random_crop \
--random_flip \
--use_fixed_image_standardization \
--learning_rate_schedule_file data/learning_rate_schedule_classifier_casia_third.txt \
--weight_decay 5e-4 \
--embedding_size 512 \
--lfw_distance_metric 1 \
--lfw_use_flipped_images \
--lfw_subtract_mean \
--validation_set_split_ratio 0.05 \
--validate_every_n_epochs 5 \
--prelogits_norm_loss_factor 5e-4


sh quant
python src/train_softmax_mobilenet.py \
--logs_base_dir ./log/facenet/ \
--models_base_dir ./model/ \
--data_dir ./dataset/casia_mtcnnpy_182/ \
--batch_size 90 \
--image_size 160 \
--lfw_dir /home/rqy/Project/data/lfw_mtcnnpy_160/ \
--pretrained_model ./model/20191101-100725/model-20191101-100725.ckpt-150 \
--optimizer ADAM \
--learning_rate -1 \
--max_nrof_epochs 150 \
--keep_probability 0.8 \
--random_crop \
--random_flip \
--use_fixed_image_standardization \
--learning_rate_schedule_file data/learning_rate_schedule_classifier_casia_third.txt \
--weight_decay 5e-4 \
--embedding_size 512 \
--lfw_distance_metric 1 \
--lfw_use_flipped_images \
--lfw_subtract_mean \
--validation_set_split_ratio 0.05 \
--validate_every_n_epochs 5 \
--prelogits_norm_loss_factor 5e-4

Saving variables
Variables saved in 0.34 seconds
Runnning forward pass on LFW images
........................
Accuracy: 0.91967+-0.01655
Validation rate: 0.46067+-0.04341 @ FAR=0.00100
Saving statistics
Epoch: [90][1/1000]     Time 0.155      Loss 3.817      Xent 2.415      RegLoss 1.403   Accuracy 0.594  Lr 0.00050      Cl 0.279
Epoch: [90][101/1000]   Time 0.200      Loss 4.485      Xent 3.081      RegLoss 1.404   Accuracy 0.375  Lr 0.00050      Cl 0.299
Epoch: [90][201/1000]   Time 0.195      Loss 5.035      Xent 3.633      RegLoss 1.402   Accuracy 0.375  Lr 0.00050      Cl 0.318
Epoch: [90][301/1000]   Time 0.194      Loss 5.070      Xent 3.669      RegLoss 1.401   Accuracy 0.469  Lr 0.00050      Cl 0.320
Epoch: [90][401/1000]   Time 0.199      Loss 4.102      Xent 2.699      RegLoss 1.403   Accuracy 0.625  Lr 0.00050      Cl 0.270
Epoch: [90][501/1000]   Time 0.169      Loss 4.702      Xent 3.300      RegLoss 1.402   Accuracy 0.469  Lr 0.00050      Cl 0.323
Epoch: [90][601/1000]   Time 0.194      Loss 3.858      Xent 2.456      RegLoss 1.402   Accuracy 0.500  Lr 0.00050      Cl 0.265
Epoch: [90][701/1000]   Time 0.187      Loss 4.404      Xent 3.001      RegLoss 1.403   Accuracy 0.375  Lr 0.00050      Cl 0.280
Epoch: [90][801/1000]   Time 0.194      Loss 4.031      Xent 2.631      RegLoss 1.400   Accuracy 0.594  Lr 0.00050      Cl 0.276
Epoch: [90][901/1000]   Time 0.129      Loss 5.756      Xent 4.355      RegLoss 1.401   Accuracy 0.281  Lr 0.00050      Cl 0.335
Running forward pass on validation set
.............................
Validation Epoch: 90    Time 37.555     Loss 5.772      Xent 4.373      Accuracy 0.287
Saving variables
Variables saved in 0.29 seconds
Runnning forward pass on LFW images
........................
Accuracy: 0.92067+-0.01679



lf 
python3 src/train_softmax_mobilenet.py \
--logs_base_dir ./log/facenet/ \
--models_base_dir ./model/ \
--data_dir ./dataset/casia_mtcnnpy_224/ \
--batch_size 64 \
--image_size 160 \
--lfw_dir ./dataset/lfw_mtcnnpy_160/ \
--pretrained_model ./model/20191031-180131/model-20191031-180131.ckpt-90 \
--optimizer ADAM \
--learning_rate -1 \
--max_nrof_epochs 150 \
--keep_probability 0.8 \
--random_crop \
--random_flip \
--use_fixed_image_standardization \
--learning_rate_schedule_file data/learning_rate_schedule_classifier_casia_second.txt \
--weight_decay 5e-4 \
--embedding_size 512 \
--lfw_distance_metric 1 \
--lfw_use_flipped_images \
--lfw_subtract_mean \
--validation_set_split_ratio 0.05 \
--validate_every_n_epochs 5 \
--prelogits_norm_loss_factor 5e-4


python3 src/train_softmax_mobilenet.py \
--logs_base_dir ./log/facenet/ \
--models_base_dir ./model/ \
--data_dir ./dataset/casia_mtcnnpy_224/ \
--batch_size 90 \
--image_size 224 \
--lfw_dir ./dataset/lfw_mtcnnpy_160/ \
--pretrained_model ./model/20191101-060848/model-20191101-060848.ckpt-249 \
--optimizer ADAM \
--learning_rate -1 \
--max_nrof_epochs 250 \
--keep_probability 0.8 \
--random_flip \
--use_fixed_image_standardization \
--learning_rate_schedule_file data/learning_rate_schedule_classifier_casia_third.txt \
--weight_decay 5e-4 \
--embedding_size 512 \
--lfw_distance_metric 1 \
--lfw_use_flipped_images \
--lfw_subtract_mean \
--validation_set_split_ratio 0.05 \
--validate_every_n_epochs 5 \
--prelogits_norm_loss_factor 5e-4


lf quant
python3 src/train_softmax_mobilenet.py \
--logs_base_dir ./log/facenet/ \
--models_base_dir ./model/ \
--data_dir ./dataset/casia_mtcnnpy_224/ \
--batch_size 32 \
--image_size 224 \
--lfw_dir ./dataset/lfw_mtcnnpy_160/ \
--pretrained_model ./model/20191104-031643/model-20191104-031643.ckpt-233 \
--optimizer ADAM \
--learning_rate -1 \
--max_nrof_epochs 250 \
--keep_probability 0.8 \
--random_flip \
--use_fixed_image_standardization \
--learning_rate_schedule_file data/learning_rate_schedule_classifier_casia_third.txt \
--weight_decay 5e-4 \
--embedding_size 512 \
--lfw_distance_metric 1 \
--lfw_use_flipped_images \
--lfw_subtract_mean \
--validation_set_split_ratio 0.05 \
--validate_every_n_epochs 5 \
--prelogits_norm_loss_factor 5e-4


----------
# ./model/20191101-100725/model-20191101-100725.ckpt-150
Saving statistics
Epoch: [149][1/1000]    Time 0.362      Loss 2.659      Xent 1.334      RegLoss 1.325   Accuracy 0.733  Lr 0.00005      Cl 0.816
Epoch: [149][101/1000]  Time 0.426      Loss 2.439      Xent 1.114      RegLoss 1.325   Accuracy 0.822  Lr 0.00005      Cl 0.785
Epoch: [149][201/1000]  Time 0.416      Loss 2.697      Xent 1.371      RegLoss 1.326   Accuracy 0.800  Lr 0.00005      Cl 0.818
Epoch: [149][301/1000]  Time 0.436      Loss 2.218      Xent 0.893      RegLoss 1.326   Accuracy 0.856  Lr 0.00005      Cl 0.755
Epoch: [149][401/1000]  Time 0.411      Loss 2.187      Xent 0.863      RegLoss 1.324   Accuracy 0.878  Lr 0.00005      Cl 0.734
Epoch: [149][501/1000]  Time 0.421      Loss 2.721      Xent 1.396      RegLoss 1.326   Accuracy 0.833  Lr 0.00005      Cl 0.792
Epoch: [149][601/1000]  Time 0.333      Loss 2.289      Xent 0.964      RegLoss 1.325   Accuracy 0.889  Lr 0.00005      Cl 0.772
Epoch: [149][701/1000]  Time 0.357      Loss 2.344      Xent 1.020      RegLoss 1.325   Accuracy 0.844  Lr 0.00005      Cl 0.769
Epoch: [149][801/1000]  Time 0.334      Loss 2.389      Xent 1.066      RegLoss 1.323   Accuracy 0.856  Lr 0.00005      Cl 0.779
Epoch: [149][901/1000]  Time 0.450      Loss 2.374      Xent 1.049      RegLoss 1.325   Accuracy 0.844  Lr 0.00005      Cl 0.793
Saving variables
Variables saved in 0.35 seconds
Runnning forward pass on LFW images
........................
Accuracy: 0.94300+-0.01513
Validation rate: 0.66067+-0.03803 @ FAR=0.00100
Saving statistics
Epoch: [150][1/1000]    Time 0.341      Loss 2.164      Xent 0.840      RegLoss 1.324   Accuracy 0.867  Lr 0.00005      Cl 0.683
Epoch: [150][101/1000]  Time 0.328      Loss 2.229      Xent 0.905      RegLoss 1.324   Accuracy 0.878  Lr 0.00005      Cl 0.717
Epoch: [150][201/1000]  Time 0.437      Loss 2.451      Xent 1.126      RegLoss 1.325   Accuracy 0.833  Lr 0.00005      Cl 0.756
Epoch: [150][301/1000]  Time 0.445      Loss 2.413      Xent 1.089      RegLoss 1.324   Accuracy 0.856  Lr 0.00005      Cl 0.779
Epoch: [150][401/1000]  Time 0.438      Loss 2.591      Xent 1.268      RegLoss 1.323   Accuracy 0.800  Lr 0.00005      Cl 0.803
Epoch: [150][501/1000]  Time 0.440      Loss 2.178      Xent 0.854      RegLoss 1.324   Accuracy 0.844  Lr 0.00005      Cl 0.712
Epoch: [150][601/1000]  Time 0.411      Loss 2.380      Xent 1.055      RegLoss 1.325   Accuracy 0.822  Lr 0.00005      Cl 0.747
Epoch: [150][701/1000]  Time 0.437      Loss 2.896      Xent 1.572      RegLoss 1.324   Accuracy 0.722  Lr 0.00005      Cl 0.803
Epoch: [150][801/1000]  Time 0.430      Loss 2.830      Xent 1.506      RegLoss 1.324   Accuracy 0.756  Lr 0.00005      Cl 0.803
Epoch: [150][901/1000]  Time 0.398      Loss 2.649      Xent 1.324      RegLoss 1.325   Accuracy 0.822  Lr 0.00005      Cl 0.797
Running forward pass on validation set
.............................
Validation Epoch: 150   Time 37.541     Loss 3.856      Xent 2.529      Accuracy 0.601
Saving variables
Variables saved in 0.28 seconds
Runnning forward pass on LFW images
........................
Accuracy: 0.94267+-0.01566
Validation rate: 0.66900+-0.04222 @ FAR=0.00100
Saving statistics


－－－－－－
20191101-060848/model-20191101-060848.ckpt-249
Epoch: [245][1/1000]    Time 0.588      Loss 3.117      Xent 1.282      RegLoss 1.835   Accuracy 0.767  Lr 0.00000      Cl 0.673
Epoch: [245][101/1000]  Time 0.606      Loss 2.829      Xent 0.994      RegLoss 1.835   Accuracy 0.867  Lr 0.00000      Cl 0.632
Epoch: [245][201/1000]  Time 0.588      Loss 3.666      Xent 1.830      RegLoss 1.836   Accuracy 0.733  Lr 0.00000      Cl 0.741
Epoch: [245][301/1000]  Time 0.599      Loss 2.662      Xent 0.825      RegLoss 1.837   Accuracy 0.889  Lr 0.00000      Cl 0.636
Epoch: [245][401/1000]  Time 0.565      Loss 3.271      Xent 1.436      RegLoss 1.835   Accuracy 0.789  Lr 0.00000      Cl 0.712
Epoch: [245][501/1000]  Time 0.582      Loss 3.035      Xent 1.199      RegLoss 1.836   Accuracy 0.844  Lr 0.00000      Cl 0.675
Epoch: [245][601/1000]  Time 0.611      Loss 2.904      Xent 1.068      RegLoss 1.835   Accuracy 0.856  Lr 0.00000      Cl 0.652
Epoch: [245][701/1000]  Time 0.607      Loss 3.057      Xent 1.222      RegLoss 1.835   Accuracy 0.844  Lr 0.00000      Cl 0.663
Epoch: [245][801/1000]  Time 0.593      Loss 2.963      Xent 1.128      RegLoss 1.835   Accuracy 0.833  Lr 0.00000      Cl 0.699
Epoch: [245][901/1000]  Time 0.636      Loss 2.629      Xent 0.795      RegLoss 1.834   Accuracy 0.900  Lr 0.00000      Cl 0.629
Running forward pass on validation set
.............................
Validation Epoch: 245   Time 77.314     Loss 4.532      Xent 2.694      Accuracy 0.572
Saving variables
Variables saved in 0.44 seconds
Runnning forward pass on LFW images
........................
Accuracy: 0.86583+-0.01948
Validation rate: 0.24467+-0.05596 @ FAR=0.00100
Saving statistics
Epoch: [246][1/1000]    Time 0.662      Loss 3.007      Xent 1.172      RegLoss 1.835   Accuracy 0.822  Lr 0.00000      Cl 0.681
Epoch: [246][101/1000]  Time 0.599      Loss 3.183      Xent 1.347      RegLoss 1.835   Accuracy 0.800  Lr 0.00000      Cl 0.722
Epoch: [246][201/1000]  Time 0.630      Loss 2.674      Xent 0.838      RegLoss 1.836   Accuracy 0.878  Lr 0.00000      Cl 0.624
Epoch: [246][301/1000]  Time 0.575      Loss 2.897      Xent 1.063      RegLoss 1.835   Accuracy 0.811  Lr 0.00000      Cl 0.676
Epoch: [246][401/1000]  Time 0.573      Loss 3.189      Xent 1.354      RegLoss 1.836   Accuracy 0.822  Lr 0.00000      Cl 0.684
Epoch: [246][501/1000]  Time 0.608      Loss 2.878      Xent 1.042      RegLoss 1.836   Accuracy 0.867  Lr 0.00000      Cl 0.655
Epoch: [246][601/1000]  Time 0.613      Loss 2.997      Xent 1.162      RegLoss 1.835   Accuracy 0.822  Lr 0.00000      Cl 0.680
Epoch: [246][701/1000]  Time 0.677      Loss 3.072      Xent 1.238      RegLoss 1.835   Accuracy 0.811  Lr 0.00000      Cl 0.661
Epoch: [246][801/1000]  Time 0.636      Loss 3.617      Xent 1.782      RegLoss 1.835   Accuracy 0.733  Lr 0.00000      Cl 0.748
Epoch: [246][901/1000]  Time 0.593      Loss 3.046      Xent 1.211      RegLoss 1.835   Accuracy 0.811  Lr 0.00000      Cl 0.653
Saving variables
Variables saved in 8.19 seconds
Runnning forward pass on LFW images
........................
Accuracy: 0.86250+-0.01781
Validation rate: 0.24600+-0.05438 @ FAR=0.00100
Saving statistics
Epoch: [247][1/1000]    Time 0.624      Loss 3.165      Xent 1.331      RegLoss 1.834   Accuracy 0.811  Lr 0.00000      Cl 0.692
Epoch: [247][101/1000]  Time 0.602      Loss 2.928      Xent 1.094      RegLoss 1.834   Accuracy 0.856  Lr 0.00000      Cl 0.653
Epoch: [247][201/1000]  Time 0.596      Loss 3.028      Xent 1.194      RegLoss 1.834   Accuracy 0.833  Lr 0.00000      Cl 0.691
Epoch: [247][301/1000]  Time 0.596      Loss 3.228      Xent 1.393      RegLoss 1.835   Accuracy 0.767  Lr 0.00000      Cl 0.699
Epoch: [247][401/1000]  Time 0.601      Loss 3.192      Xent 1.357      RegLoss 1.835   Accuracy 0.789  Lr 0.00000      Cl 0.703
Epoch: [247][501/1000]  Time 0.641      Loss 3.147      Xent 1.311      RegLoss 1.836   Accuracy 0.778  Lr 0.00000      Cl 0.686
Epoch: [247][601/1000]  Time 0.610      Loss 2.848      Xent 1.014      RegLoss 1.834   Accuracy 0.844  Lr 0.00000      Cl 0.662
Epoch: [247][701/1000]  Time 0.593      Loss 3.070      Xent 1.236      RegLoss 1.834   Accuracy 0.844  Lr 0.00000      Cl 0.642
Epoch: [247][801/1000]  Time 0.602      Loss 3.000      Xent 1.166      RegLoss 1.834   Accuracy 0.822  Lr 0.00000      Cl 0.661
Epoch: [247][901/1000]  Time 0.635      Loss 2.846      Xent 1.011      RegLoss 1.836   Accuracy 0.833  Lr 0.00000      Cl 0.647
Saving variables
Variables saved in 0.59 seconds
Runnning forward pass on LFW images
........................
Accuracy: 0.86050+-0.01773
Validation rate: 0.24667+-0.05395 @ FAR=0.00100
Saving statistics
Epoch: [248][1/1000]    Time 0.635      Loss 2.595      Xent 0.762      RegLoss 1.834   Accuracy 0.878  Lr 0.00000      Cl 0.589
Epoch: [248][101/1000]  Time 0.606      Loss 2.905      Xent 1.071      RegLoss 1.835   Accuracy 0.856  Lr 0.00000      Cl 0.620
Epoch: [248][201/1000]  Time 0.605      Loss 2.806      Xent 0.971      RegLoss 1.836   Accuracy 0.833  Lr 0.00000      Cl 0.635
Epoch: [248][301/1000]  Time 0.602      Loss 2.697      Xent 0.861      RegLoss 1.836   Accuracy 0.878  Lr 0.00000      Cl 0.603
Epoch: [248][401/1000]  Time 0.603      Loss 2.868      Xent 1.032      RegLoss 1.836   Accuracy 0.889  Lr 0.00000      Cl 0.635
Epoch: [248][501/1000]  Time 0.600      Loss 3.367      Xent 1.533      RegLoss 1.834   Accuracy 0.778  Lr 0.00000      Cl 0.679
Epoch: [248][601/1000]  Time 0.602      Loss 2.886      Xent 1.050      RegLoss 1.836   Accuracy 0.856  Lr 0.00000      Cl 0.666
Epoch: [248][701/1000]  Time 0.589      Loss 2.989      Xent 1.155      RegLoss 1.834   Accuracy 0.789  Lr 0.00000      Cl 0.645
Epoch: [248][801/1000]  Time 0.606      Loss 3.121      Xent 1.286      RegLoss 1.835   Accuracy 0.811  Lr 0.00000      Cl 0.679
Epoch: [248][901/1000]  Time 0.616      Loss 2.724      Xent 0.889      RegLoss 1.835   Accuracy 0.844  Lr 0.00000      Cl 0.619
Saving variables
Variables saved in 0.56 seconds
Runnning forward pass on LFW images
........................
Accuracy: 0.86467+-0.01810
Validation rate: 0.24733+-0.05341 @ FAR=0.00100
Saving statistics
Epoch: [249][1/1000]    Time 0.604      Loss 2.781      Xent 0.948      RegLoss 1.834   Accuracy 0.878  Lr 0.00000      Cl 0.601
Epoch: [249][101/1000]  Time 0.600      Loss 2.783      Xent 0.948      RegLoss 1.835   Accuracy 0.856  Lr 0.00000      Cl 0.633
Epoch: [249][201/1000]  Time 0.584      Loss 2.837      Xent 1.001      RegLoss 1.836   Accuracy 0.800  Lr 0.00000      Cl 0.688
Epoch: [249][301/1000]  Time 0.612      Loss 2.887      Xent 1.052      RegLoss 1.836   Accuracy 0.856  Lr 0.00000      Cl 0.671
Epoch: [249][401/1000]  Time 0.595      Loss 2.940      Xent 1.105      RegLoss 1.835   Accuracy 0.811  Lr 0.00000      Cl 0.700
Epoch: [249][501/1000]  Time 0.600      Loss 3.024      Xent 1.189      RegLoss 1.835   Accuracy 0.811  Lr 0.00000      Cl 0.672
Epoch: [249][601/1000]  Time 0.607      Loss 2.984      Xent 1.148      RegLoss 1.836   Accuracy 0.844  Lr 0.00000      Cl 0.701
Epoch: [249][701/1000]  Time 0.584      Loss 2.890      Xent 1.054      RegLoss 1.836   Accuracy 0.800  Lr 0.00000      Cl 0.631
Epoch: [249][801/1000]  Time 0.602      Loss 2.894      Xent 1.057      RegLoss 1.836   Accuracy 0.822  Lr 0.00000      Cl 0.647
Epoch: [249][901/1000]  Time 0.568      Loss 2.749      Xent 0.914      RegLoss 1.836   Accuracy 0.844  Lr 0.00000      Cl 0.604
Saving variables
Variables saved in 0.48 seconds
Runnning forward pass on LFW images


--------------
sh 20191106-110538
Accuracy: 0.94283+-0.01461
Validation rate: 0.66533+-0.04003 @ FAR=0.00100
Saving statistics
Epoch: [150][1/1000]    Time 0.361      Loss 2.126      Xent 0.819      RegLoss 1.307   Accuracy 0.911  Lr 0.00000      Cl 0.677
Epoch: [150][101/1000]  Time 0.415      Loss 2.267      Xent 0.959      RegLoss 1.308   Accuracy 0.844  Lr 0.00000      Cl 0.734
Epoch: [150][201/1000]  Time 0.447      Loss 2.417      Xent 1.109      RegLoss 1.308   Accuracy 0.833  Lr 0.00000      Cl 0.760
Epoch: [150][301/1000]  Time 0.421      Loss 2.347      Xent 1.039      RegLoss 1.308   Accuracy 0.867  Lr 0.00000      Cl 0.782
Epoch: [150][401/1000]  Time 0.410      Loss 2.562      Xent 1.256      RegLoss 1.306   Accuracy 0.789  Lr 0.00000      Cl 0.807
Epoch: [150][501/1000]  Time 0.337      Loss 2.178      Xent 0.871      RegLoss 1.307   Accuracy 0.900  Lr 0.00000      Cl 0.732
Epoch: [150][601/1000]  Time 0.427      Loss 2.158      Xent 0.851      RegLoss 1.307   Accuracy 0.867  Lr 0.00000      Cl 0.739
Epoch: [150][701/1000]  Time 0.333      Loss 2.824      Xent 1.516      RegLoss 1.308   Accuracy 0.733  Lr 0.00000      Cl 0.800
Epoch: [150][801/1000]  Time 0.426      Loss 2.817      Xent 1.510      RegLoss 1.308   Accuracy 0.733  Lr 0.00000      Cl 0.809
Epoch: [150][901/1000]  Time 0.408      Loss 2.683      Xent 1.374      RegLoss 1.308   Accuracy 0.800  Lr 0.00000      Cl 0.800
Running forward pass on validation set
.............................
Validation Epoch: 150   Time 37.689     Loss 3.825      Xent 2.515      Accuracy 0.606
Saving variables
Variables saved in 0.29 seconds


-------------
lf 20191104-031643/

Epoch: [232][1/1000]    Time 0.690      Loss 2.877      Xent 1.076      RegLoss 1.801   Accuracy 0.856  Lr 0.00000      Cl 0.702
Epoch: [232][101/1000]  Time 0.610      Loss 2.818      Xent 1.020      RegLoss 1.798   Accuracy 0.833  Lr 0.00000      Cl 0.658
Epoch: [232][201/1000]  Time 0.606      Loss 2.737      Xent 0.939      RegLoss 1.798   Accuracy 0.822  Lr 0.00000      Cl 0.656
Epoch: [232][301/1000]  Time 0.603      Loss 2.977      Xent 1.178      RegLoss 1.799   Accuracy 0.833  Lr 0.00000      Cl 0.680
Epoch: [232][401/1000]  Time 0.605      Loss 2.958      Xent 1.160      RegLoss 1.798   Accuracy 0.833  Lr 0.00000      Cl 0.693
Epoch: [232][501/1000]  Time 0.599      Loss 3.283      Xent 1.485      RegLoss 1.798   Accuracy 0.778  Lr 0.00000      Cl 0.705
Epoch: [232][601/1000]  Time 0.645      Loss 2.898      Xent 1.101      RegLoss 1.797   Accuracy 0.856  Lr 0.00000      Cl 0.673

Epoch: [232][701/1000]  Time 0.578      Loss 3.046      Xent 1.247      RegLoss 1.799   Accuracy 0.856  Lr 0.00000      Cl 0.665
Epoch: [232][801/1000]  Time 0.577      Loss 2.592      Xent 0.794      RegLoss 1.798   Accuracy 0.844  Lr 0.00000      Cl 0.598
Epoch: [232][901/1000]  Time 0.620      Loss 2.892      Xent 1.092      RegLoss 1.799   Accuracy 0.856  Lr 0.00000      Cl 0.670
Saving variables
Variables saved in 0.47 seconds
Runnning forward pass on LFW images
........................
Accuracy: 0.86150+-0.01843
Validation rate: 0.24367+-0.04202 @ FAR=0.00100
Saving statistics
Epoch: [233][1/1000]    Time 0.610      Loss 2.658      Xent 0.860      RegLoss 1.798   Accuracy 0.878  Lr 0.00000      Cl 0.604
Epoch: [233][101/1000]  Time 0.577      Loss 2.844      Xent 1.044      RegLoss 1.800   Accuracy 0.878  Lr 0.00000      Cl 0.637
Epoch: [233][201/1000]  Time 0.662      Loss 2.984      Xent 1.186      RegLoss 1.798   Accuracy 0.856  Lr 0.00000      Cl 0.659
Epoch: [233][301/1000]  Time 0.636      Loss 2.776      Xent 0.978      RegLoss 1.798   Accuracy 0.844  Lr 0.00000      Cl 0.646
Epoch: [233][401/1000]  Time 0.691      Loss 2.707      Xent 0.907      RegLoss 1.799   Accuracy 0.889  Lr 0.00000      Cl 0.660
Epoch: [233][501/1000]  Time 0.597      Loss 2.542      Xent 0.745      RegLoss 1.798   Accuracy 0.878  Lr 0.00000      Cl 0.623
Epoch: [233][601/1000]  Time 0.608      Loss 2.950      Xent 1.150      RegLoss 1.800   Accuracy 0.867  Lr 0.00000      Cl 0.677
Epoch: [233][701/1000]  Time 0.607      Loss 2.834      Xent 1.036      RegLoss 1.798   Accuracy 0.867  Lr 0.00000      Cl 0.664
Epoch: [233][801/1000]  Time 0.612      Loss 3.015      Xent 1.217      RegLoss 1.798   Accuracy 0.811  Lr 0.00000      Cl 0.684
Epoch: [233][901/1000]  Time 0.609      Loss 2.882      Xent 1.082      RegLoss 1.800   Accuracy 0.856  Lr 0.00000      Cl 0.679
Saving variables
Variables saved in 0.94 seconds
Runnning forward pass on LFW images
........................
Accuracy: 0.86217+-0.01821
Validation rate: 0.22567+-0.04867 @ FAR=0.00100
Saving statistics
Epoch: [234][1/1000]    Time 0.644      Loss 2.838      Xent 1.039      RegLoss 1.799   Accuracy 0.867  Lr 0.00000      Cl 0.691
Epoch: [234][101/1000]  Time 0.615      Loss 3.006      Xent 1.206      RegLoss 1.800   Accuracy 0.833  Lr 0.00000      Cl 0.722
Epoch: [234][201/1000]  Time 0.604      Loss 2.626      Xent 0.827      RegLoss 1.799   Accuracy 0.867  Lr 0.00000      Cl 0.638
Epoch: [234][301/1000]  Time 0.565      Loss 2.833      Xent 1.035      RegLoss 1.798   Accuracy 0.822  Lr 0.00000      Cl 0.680
Epoch: [234][401/1000]  Time 0.658      Loss 3.097      Xent 1.298      RegLoss 1.799   Accuracy 0.822  Lr 0.00000      Cl 0.707



---------
triplet loss 
first training
python3 src/train_tripletloss_mobilenet.py --logs_base_dir ./log/facenet/ --models_base_dir ./model/ --data_dir ./dataset/casia_mtcnnpy_224/ --batch_size 90 --image_size 224 --lfw_dir ./dataset/lfw_mtcnnpy_160/ --optimizer ADAM --learning_rate -1 --max_nrof_epochs 250 --keep_probability 0.8 --random_flip --use_fixed_image_standardization --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt--weight_decay 5e-4 --embedding_size 512 --lfw_distance_metric 1 --lfw_use_flipped_images --lfw_subtract_mean
20191113-065711/model-20191113-065711

python3 src/train_tripletloss_mobilenet.py --pretrained_model ./model/20191113-065711/model-20191113-065711.ckpt-79033 --logs_base_dir ./log/facenet/ --models_base_dir ./model/ --data_dir ./dataset/casia_mtcnnpy_224/ --batch_size 90 --image_size 224 --lfw_dir ./dataset/lfw_mtcnnpy_160/ --optimizer ADAM --learning_rate -1 --max_nrof_epochs 250 --keep_probability 0.8 --random_flip --use_fixed_image_standardization --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia_second.txt --weight_decay 5e-4 --embedding_size 512 --lfw_distance_metric 1 --lfw_use_flipped_images --lfw_subtract_mean

 20191114-031120/
Epoch: [248][1201/1000] Time 0.535      Loss 0.131
Epoch: [248][1401/1000] Time 0.607      Loss 0.139
Saving variables
Variables saved in 0.09 seconds
Running forward pass on LFW images: 82.690
Accuracy: 0.803+-0.015
Validation rate: 0.09233+-0.01687 @ FAR=0.00100

python3 src/train_tripletloss_mobilenet.py --pretrained_model ./model/20191114-031120/model-20191114-031120.ckpt-246603 --logs_base_dir ./log/facenet/ --models_base_dir ./model/ --data_dir ./dataset/casia_mtcnnpy_224/ --batch_size 90 --image_size 224 --lfw_dir ./dataset/lfw_mtcnnpy_160/ --optimizer ADAM --learning_rate -1 --max_nrof_epochs 250 --keep_probability 0.8 --random_flip --use_fixed_image_standardization --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia_second.txt --weight_decay 5e-4 --embedding_size 512 --lfw_distance_metric 1 --lfw_use_flipped_images --lfw_subtract_mean


python3 src/train_tripletloss_mobilenet.py --pretrained_model ./model/20191113-065711/model-20191113-065711.ckpt-79033 --logs_base_dir ./log/facenet/ --models_base_dir ./model/ --data_dir ./dataset/casia_mtcnnpy_224/ --batch_size 90 --image_size 224 --lfw_dir ./dataset/lfw_mtcnnpy_160/ --optimizer ADAM --learning_rate -1 --max_nrof_epochs 250 --keep_probability 0.8 --random_flip --use_fixed_image_standardization --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia_third.txt --weight_decay 5e-4 --embedding_size 512 --lfw_distance_metric 1 --lfw_use_flipped_images --lfw_subtract_mean

