from easydict import EasyDict

config = EasyDict()
# if caffe_use=1, face_model.py file will load caffe model to generate face features for face recognition
config.caffe_use = 0
# The value of feature_lenth is 512 or 256. The length of the feature is determined by the dimension of the feature
config.feature_lenth = 512 #256#1024
# to load the insightface caffemodel
config.insight = 1
# used to predict the class of the face. only used for  DB_pred function in reg_test.py.
config.label = 1
# if torch_=1, will load the pytorch model
config.torch_ = 0
#if mx_=1, will load the mxnet model
config.mx_ = 1
#if debug=1,will pring information among  all the face reg files
config.debug = 0
#in mxnet,used in face_model.py, mx_version to select the method of loading the mxnet model
config.mx_version = 1
#used in face_model.py, norm used to whether normolize the features output. only used for mxnet
config.norm = 0
#only used for mxnet and caffemodel, if feature_expand=1, the input of the mxnet net will be 2 images, and the output features of the face will add the 2 images features. 
config.feature_expand = 1
# whether use gamma enhancement for database images
config.db_enhance = 0
# the subtraction of  top 2 distances, threshold value
config.confidence = 0.1
# the nearest person, distance threshold value
config.top1_distance = 1.1
# whether to load center_loss caffemodel
config.center_loss= 0
# whether to use image enhance to process the faces for face recognition
config.face_enhance = 0
# whether to print the detect and recognize consuming time
config.time = 0
# after load caffe models successful, resaving the caffemodel
config.model_resave = 0
#frame interval to recognize
config.frame_interval = 50
#extract the 1024 feature
config.feature_1024 = 0
#image enhance gamma
config.gamma = 0.5
#for face detect fun, the output box whether widen
config.box_widen = 0
#whether to run face detect
config.face_detect = 0
#whether the img saved dir name is 11****   1:6666; 0: 116666
config.dir_label = 0
#whether to show the images
config.show_img = 0
#whether to save wrong recognize imgs 
config.save_failedimg = 1
#whether to save id images recognized
config.save_idimg = 1
#if the input images has label
config.reg_fromlabel = 1
# if pass the lenth of image name less 8
config.img_lenth8 = 1
#using frame_num to control recognition result
config.use_framenum = 0