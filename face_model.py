# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/07/2 14:09
#project: Face recognize
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  caffe and tensorflow
####################################################
import sys
sys.path.append('../')
#sys.path.append('/home/lxy/caffe/python')
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
import cv2
import numpy as np
import time
from scipy.spatial import distance
from face_config import config
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