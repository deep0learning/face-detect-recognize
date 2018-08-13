import numpy as np 
from easydict import EasyDict

config = EasyDict()
# if caffe_use=1, face_model.py file will load caffe model to generate face features for face recognition
config.caffe_use = 1
# The value of feature_lenth is 512 or 256. The length of the feature is determined by the dimension of the feature
config.feature_lenth = 512 #256
# to load the insightface caffemodel
config.insight = 1
# used to predict the class of the face. only used for  DB_pred function in reg_test.py.
config.label = 1
# if torch_=1, will load the pytorch model
config.torch_ = 0
#if mx_=1, will load the mxnet model
config.mx_ = 0
#if debug=1,will pring information among  all the face reg files
config.debug = 0
#in mxnet,used in face_model.py, mx_version to select the method of loading the mxnet model
config.mx_version = 1
#used in face_model.py, norm used to whether normolize the features output. only used for mxnet
config.norm = 0
#only used for mxnet and caffemodel, if feature_expand=1, the input of the mxnet net will be 2 images, and the output features of the face will add the 2 images features. 
config.feature_expand = 0
# whether use gamma enhancement for database images
config.db_enhance = 0
# the subtraction of  top 2 distances, threshold value
config.confidence = 0.11
# the nearest person, distance threshold value
config.top1_distance =1.2
# whether to load center_loss caffemodel
config.center_loss= 0
# whether to use image enhance to process the faces for face recognition
config.face_enhance = 1
# whether to print the detect and recognize consuming time
config.time = 0