# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/07/2 14:09
#project: Face recognize
#company: Senscape
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  caffe and tensorflow
####################################################
import sys
sys.path.append('../')
#sys.path.append('/home/lxy/caffe/python')
sys.path.append('/home/lxy/Develop/Center_Loss/sphereface/tools/caffe-sphereface/python')
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
import cv2
import numpy as np
import time
from scipy.spatial import distance
from face_config import config
#from L_Resnet_E_IR import get_resnet
#import tensorflow as tf
#import torch
#from ResNet import ResNet50M
#from torchvision import transforms as tt
import mxnet as mx
from sklearn import preprocessing as skpro

def L2_distance(feature1,feature2,lenth):
    print("feature shape: ",np.shape(feature1))
    feature1 = np.asarray(feature1)
    feature2 = np.asarray(feature2)
    len1 = feature1.shape[0]
    len2 = feature2.shape[0]
    assert len1 ==lenth and len2==lenth
    mean1 = np.mean(feature1)
    mean2 = np.mean(feature2)
    print("feature mean: ",mean1, mean2)
    f_center1 = feature1-mean1
    f_center2 = feature2-mean2
    std1 = np.sum(np.power(f_center1,2))
    std2 = np.sum(np.power(f_center2,2))
    std1 = np.sqrt(std1)
    std2 = np.sqrt(std2)
    norm1 = f_center1/std1
    norm2 = f_center2/std2
    loss =np.sqrt(np.sum(np.power((norm1-norm2),2)))
    return loss


class FaceReg(object):
    def __init__(self,threshold):
        self.threshold = threshold
        self.load_model()
        caffe.set_device(0)
        caffe.set_mode_gpu()
    def load_model(self):
        if config.center_loss:
            face_p = "../models/center/face.prototxt"
            face_m = "../models/center/face.caffemodel"
        else:
            #face_p = "../models/sphere/sph_2.prototxt"
            #face_m = "../models/sphere/sph_2.caffemodel"
            #face_m = "../models/model-r50-am.caffemodel"
            #face_p = "../models/center/center_loss.prototxt"
            face_p = "../models/sphere/sph_2.prototxt"
            if config.feature_1024:
                face_p = "../models/sphere/sph_1.prototxt"
            #face_m = "../models/sphere/sph20_10811.caffemodel" #good
            #face_m = "../models/sphere/sph20_ms_v4.caffemodel" #very good
            #face_m = "../models/sphere/sph20_ms_v6.caffemodel" #very good
            #face_m = "../models/sphere/sph20_ms_v7.caffemodel"  #best good
            #face_m = "../models/sphere/sph20_ms_v8.caffemodel"  #the best good
            #face_m = "../models/sphere/sph20_ms_v9.caffemodel"   #fpr is best good
            #face_m = "../models/sphere/sph20_ms_4v5.caffemodel"  # **the best
            #face_m = "../models/sphere/sph20_ms_1024v1.caffemodel" #best for m=1
            #face_m = "../models/sphere/sph_test.caffemodel"
            face_m = "../models/sphere/sph20_ms_5v2.caffemodel"
            #face_m = "../models/sphere/sph20_ms_6v3.caffemodel" good for m=1
            #face_m = "../models/sphere/sph20_ms_8v1.caffemodel"
            #face_m = "../models/sphere/sph20_ms_8v3.caffemodel" good for m=1
            #face_p = "../models/sphere/sph_64_model.prototxt"
            #face_m = "../models/sphere/sph64_10811.caffemodel"
            #face_m = "../models/sphere/sph64_ms_v1.caffemodel"
        self.face_net = caffe.Net(face_p,face_m,caffe.TEST)
        print("face model load successful **************************************")
        if config.caffe_resave:
            caffe_model_path = "../models/sphere/sph_test.caffemodel"
            self.face_net.save(caffe_model_path)
            print("************ model resaved over******")
    def extractfeature(self,img):
        h,w,chal = img.shape
        #print(img.shape)
        #print("img shape ",img.shape)
        #img = np.expandim(img,0)
        _,net_chal,net_h,net_w = self.face_net.blobs['data'].data.shape
        #print("net shape ",net_h,net_w)
        if h !=net_h or w !=net_w:
            caffe_img = cv2.resize(img,(net_w,net_h))
        else:
            caffe_img = img
        if config.feature_expand:
            self.face_net.blobs['data'].reshape(2,net_chal,net_h,net_w)
        else:
            self.face_net.blobs['data'].reshape(1,net_chal,net_h,net_w)
        t = time.time()
        #caffe_img = np.swapaxes(caffe_img,0,2)
        caffe_img = np.expand_dims(caffe_img,0)
        caffe_img = np.transpose(caffe_img, (0,3,1,2))
        if config.feature_expand:
            img_flip = np.flip(caffe_img,axis=3)
            self.face_net.blobs['data'].data[0] = caffe_img
            self.face_net.blobs['data'].data[1] = img_flip
        else:
            self.face_net.blobs['data'].data[...] = caffe_img
        net_out = self.face_net.forward()
        if config.feature_1024:
            features = net_out['fc5_n']
        else:
            features = net_out['fc5']
        t1 = time.time()-t
        if config.time:
            print("caffe forward time cost: ",t1)
        if config.feature_expand:
            feature_org = features[0]
            feature_flip = features[1]
            feature_sum = feature_org + feature_flip
            f_mean = np.mean(feature_sum)
            f_std = np.std(feature_sum)
            features = (feature_sum - f_mean)/f_std
            #f_min = np.min(feature_sum)
            #f_max = np.max(feature_sum)
            #features = (feature_sum - f_min)/(f_max-f_min)
        return np.reshape(features,(config.feature_lenth))
    def prob_cls(self,img):
        h,w,chal = img.shape
        #print("img shape ",img.shape)
        _,net_chal,net_h,net_w = self.face_net.blobs['data'].data.shape
        #print("net shape ",net_h,net_w)
        if h !=net_h or w !=net_w:
            caffe_img = cv2.resize(img,(net_w,net_h))
        else:
            caffe_img = img
        '''
        dist_img = np.zeros([3,net_h,net_w])
        dist_img[0,:,:] = caffe_img[:,:,0]
        dist_img[1,:,:] = caffe_img[:,:,1]
        dist_img[2,:,:] = caffe_img[:,:,2]
        caffe_img = dist_img
        '''
        t = time.time()
        #caffe_img = np.swapaxes(caffe_img,0,2)
        caffe_img = np.expand_dims(caffe_img,0)
        caffe_img = np.transpose(caffe_img, (0,3,1,2))
        self.face_net.blobs['data'].data[...] = caffe_img
        if config.label:
            self.face_net.blobs['label'].data[...] = 1
        net_out = self.face_net.forward()
        if config.insight:
            pred = net_out['prob1']
        else:
            pred = net_out['fc5']
        t1 = time.time()-t
        #print("caffse forward time cost: ",t1)
        #return np.array(features)
        return pred
    def calculateL2(self,feat1,feat2,c_type='euclidean'):
        assert np.shape(feat1)==np.shape(feat2)
        if config.insight:
            [len_,]= np.shape(feat1)
            #print(np.shape(feat1))
        else:
            _,len_ = np.shape(feat1)
        #print("len ",len_)
        if c_type == "cosine":
            s_d = distance.cosine(feat1,feat2)
        elif c_type == "euclidean":
            #s_d = np.sqrt(np.sum(np.square(feat1-feat2)))
            s_d = distance.euclidean(feat1,feat2,w=1./len_)
            #s_d = distance.euclidean(feat1,feat2,w=1)
        elif c_type == "correlation":
            s_d = distance.correlation(feat1,feat2)
        elif c_type == "braycurtis":
            s_d = distance.braycurtis(feat1,feat2)
        elif c_type == 'canberra':
            s_d = distance.canberra(feat1,feat2)
        elif c_type == "chebyshev":
            s_d = distance.chebyshev(feat1,feat2)
        return s_d

class TF_Face_Reg(object):
    def __init__(self,model_file,img_size,gpu_num):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
        graph = tf.Graph()
        self.h,self.w = img_size
        with graph.as_default():
            self.image_op = tf.placeholder(name="img_inputs",shape=[1, self.h,self.w, 3], dtype=tf.float32)
            labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
            w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
            self.net = get_resnet(self.image_op,50,type='ir', w_init=w_init_method, trainable=False)
            #logit = arcface_loss(embedding=self.net.outputs, labels=labels, w_init=w_init_method, out_num=85164)
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth=True  
            tf_config.log_device_placement=False
            self.sess = tf.Session(config=tf_config)
            #saver = tf.train.import_meta_graph('../models/tf-models/InsightFace_iter_350000.meta')
            saver = tf.train.Saver(max_to_keep=100)
            saver.restore(self.sess,model_file)
            all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            #for var_ in all_vars:
                #print(var_)
            for v_name in tf.global_variables():
                print("name : ",v_name.name[:-2],v_name.shape) 

    def extractfeature(self,img):
        h_,w_,chal_ = img.shape
        #print("img shape ",img.shape)
        #img = np.expandim(img,0)
        if h_ !=self.h or w_ !=self.w:
            tf_img = cv2.resize(img,(self.w,self.h))
        else:
            tf_img = img
        caffe_img = np.expand_dims(tf_img,0)
        feat = self.sess.run([self.net.outputs],feed_dict={self.image_op:caffe_img})
        return np.array(feat)
    def calculateL2(self,feat1,feat2,c_type='euclidean'):
        assert np.shape(feat1)==np.shape(feat2)
        len_ = np.shape(feat1)
        #print("len ",len_)
        if c_type == "cosine":
            s_d = distance.cosine(feat1,feat2)
        elif c_type == "euclidean":
            s_d = distance.euclidean(feat1,feat2,w=1./len_[-1])
        elif c_type == "correlation":
            s_d = distance.correlation(feat1,feat2)
        elif c_type == "braycurtis":
            s_d = distance.braycurtis(feat1,feat2)
        elif c_type == 'canberra':
            s_d = distance.canberra(feat1,feat2)
        elif c_type == "chebyshev":
            s_d = distance.chebyshev(feat1,feat2)
        return s_d

class Deep_Face(object):
    def __init__(self,model_path,h,w):
        self.h = h
        self.w = w
        self.model_path = model_path
        self.load_model()
    def load_model(self):
        self.model_net = ResNet50M(num_class=210,loss={'xent'},use_gpu=True)
        checkpoint = torch.load(self.model_path)
        self.model_net.load_state_dict(checkpoint['state_dict'])
    def extractfeature(self,img):
        h_,w_,chal_ = img.shape
        if h_ !=self.h or w_ !=self.w:
            img = cv2.resize(img,(self.w,self.h))
        img[:,:,0] = (img[:,:,0]-0.485)/0.229
        img[:,:,1] = (img[:,:,1]-0.456)/0.224
        img[:,:,2] = (img[:,:,2]-0.406)/0.225
        img = np.transpose(img,(2,0,1))
        #img = np.expand_dims(img,0)
        tor_img = torch.from_numpy(img)
        tor_img = tor_img.cuda()
        self.model_net.eval()
        feat = self.model_net(tor_img)
        return feat.numpy()

class mx_Face(object):
    def __init__(self,model_path,epoch_num,img_size,layer='fc1'):
        ctx = mx.gpu()
        if config.mx_version:
            self.model_net = self.load_model2(ctx,img_size,model_path,epoch_num,layer)
        else:
            self.load_model(model_path,epoch_num)
        self.h, self.w = img_size

    def load_model(self,model_path,epoch_num):
        #sym,arg_params,aux_params = mx.model.load_checkpoint(model_path,epoch)
        #mod_net mx.mod.Module()
        self.model_net = mx.model.FeedForward.load(model_path,epoch_num,ctx=mx.gpu())

    def load_model2(self,ctx, image_size, prefix,epoch_num,layer):
        print('loading',prefix, epoch_num)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch_num)
        all_layers = sym.get_internals()
        if config.debug:
            print("all layers ",all_layers.list_outputs()[-1])
        sym = all_layers[layer+'_output']
        model = mx.mod.Module(symbol=sym, context=ctx, data_names=('data',),label_names = None)
        #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
        if config.feature_expand:
            model.bind(data_shapes=[('data', (2, 3, image_size[0], image_size[1]))])
        else:
            model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))],for_training=False)
        model.set_params(arg_params, aux_params)
        return model

    def extractfeature_(self,img):
        img_list = []
        h_,w_,chal_ = img.shape
        if h_ !=self.h or w_ !=self.w:
            img = cv2.resize(img,(self.w,self.h))
        img = (img-127.5)*0.0078125
        img = np.transpose(img,(2,0,1))
        img_flip = np.flip(img,axis=2)
        #img = np.expand_dims(img,0)
        img_list.append(img)
        #img_flip = np.expand_dims(img_flip,0)
        img_list.append(img_flip)
        if config.mx_version:
            data = mx.nd.array(img_list)
            db = mx.io.DataBatch(data=(data,))
            self.model_net.forward(db, is_train=False)
            embedding = self.model_net.get_outputs()[0].asnumpy()
            features = embedding[0] + embedding[1]
            features = np.expand_dims(features,0)
            features = skpro.normalize(features)
        else: 
            features = self.model_net.predict(img)
            #embedding = features[0]
        if config.debug:
            print("feature shape ",np.shape(features))
        #print("features ",features[0,:5])
        return features[0]

    def extractfeature(self,img):
        h_,w_,chal_ = img.shape
        if h_ !=self.h or w_ !=self.w:
            img = cv2.resize(img,(self.w,self.h))
        img = (img-127.5)*0.0078125
        img = np.transpose(img,(2,0,1))
        img = np.expand_dims(img,0)
        if config.mx_version:
            data = mx.nd.array(img)
            db = mx.io.DataBatch(data=(data,))
            self.model_net.forward(db, is_train=False)
            embedding = self.model_net.get_outputs()[0].asnumpy()
            #embedding = skpro.normalize(embedding)
            features = embedding 
        else: 
            features = self.model_net.predict(img)
            #embedding = features[0]
        if config.debug:
            print("feature shape ",np.shape(features))
        #print("features ",features[0,:5])
        return features[0]

    def calculateL2(self,feat1,feat2,c_type='euclidean'):
        assert np.shape(feat1)==np.shape(feat2)
        if config.insight:
            [len_,]= np.shape(feat1)
            #print(np.shape(feat1))
        else:
            _,len_ = np.shape(feat1)
        #print("len ",len_)
        if c_type == "cosine":
            s_d = distance.cosine(feat1,feat2)
        elif c_type == "euclidean":
            #s_d = np.sqrt(np.sum(np.square(feat1-feat2)))
            s_d = distance.euclidean(feat1,feat2,w=1./len_)
            #s_d = distance.euclidean(feat1,feat2,w=1)
        elif c_type == "correlation":
            s_d = distance.correlation(feat1,feat2)
        elif c_type == "braycurtis":
            s_d = distance.braycurtis(feat1,feat2)
        elif c_type == 'canberra':
            s_d = distance.canberra(feat1,feat2)
        elif c_type == "chebyshev":
            s_d = distance.chebyshev(feat1,feat2)
        return s_d