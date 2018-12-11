# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/06/14 14:09
#project: Face recognize
#company: Senscape
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  test caffe model and figure confusionMatrix
####################################################

import sys
sys.path.append('../')
import os
os.environ['GLOG_minloglevel'] = '2'
from demo.Detector import MTCNNDet
import cv2
import numpy as np
import argparse
import time
from scipy.spatial import distance,cKDTree
from face_config import config
from face_model import FaceReg,TF_Face_Reg,Deep_Face,mx_Face,L2_distance,L2_distance_
import h5py
import shutil
from annoy import AnnoyIndex
import pickle
from extract_features import ExtractFeatures


def args():
    parser = argparse.ArgumentParser(description="mtcnn caffe")
    parser.add_argument('--file-in',type=str,dest='file_in',default='None',\
                        help="the file input path")
    parser.add_argument('--img-path1',type=str,dest='img_path1',default="test1.jpg",\
                        help="img1 saved path")
    parser.add_argument('--img-path2',type=str,dest='img_path2',default="test2.jpg",\
                        help="scale img size")
    parser.add_argument('--base-dir',type=str,dest='base_dir',default="./base_dir",\
                        help="images saved dir")
    parser.add_argument('--out_file', default='./output/L1_regular.txt', type=str,help='save txt')
    parser.add_argument('--cmd-type', default="dbtest",dest='cmd_type', type=str,\
                        help="which code to run: evaluate,imgtest ")
    return parser.parse_args()

def write_hdf5(file, data, label_class):
    # transform to np array
    data_arr = np.array(data, dtype = np.float32)
    # if no swapaxes, transpose to num * channel * width * height ???
    # data_arr = data_arr.transpose(0, 3, 2, 1)
    label_class_arr = np.array(label_class, dtype = np.float32)
    with h5py.File(file, 'w') as f:
        f['data'] = data_arr
        f['label_class'] = label_class_arr
        f.close()

def read_img(path_img):
    img = cv2.imread(path_img)
    img_org = img
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = (img-127.5)*0.0078125
    return img,img_org

def save_mat(caffe_img,file_name):
    print(caffe_img[0,1,:])
    img_r = np.reshape(caffe_img[:,:,0],(-1,1))
    img_g = np.reshape(caffe_img[:,:,1],(-1,1))
    img_b = np.reshape(caffe_img[:,:,2],(-1,1))
    print("img ",np.shape(img_b))
    rgb = np.concatenate((img_r,img_g,img_b),axis=1)
    print(rgb.shape)
    np.savetxt(file_name,rgb)

def L1_Sum(feature):
    fe_abs = np.abs(feature)
    return np.sum(fe_abs)

def compare_img(path1,path2):
    '''
    path1: image1 path
    path2: saved path of image2
    return: L1_regular of features from image1 and image2
    '''
    img1,img_1 = read_img(path1)
    img2,img_2 = read_img(path2)
    if config.mx_ :
        #model_path = "../models/mx_models/v1_bn/model"
        #model_path = "/home/lxy/Develop/Center_Loss/arcface/insightface/models/model-r100-ii/model"
        #model_path = "/home/lxy/Develop/Center_Loss/arcface/insightface/models/model-y1-test2/model"
        #model_path = "../models/mx_models/mobile_model/model"
        model_path = "../models/mx_models/model-100/model-v3/modelresave"
        #model_path = "../models/mx_models/model-100/model-org/modelresave"
        epoch_num = 0  #9#2
        img_size = [112,112]
        FaceModel = mx_Face(model_path,epoch_num,img_size)
    elif config.caffe_use:
        #face_p = "../models/sphere/sph_2.prototxt"
        #face_m = "../models/sphere/sph20_ms_4v5.caffemodel"
        face_p = "../models/mx_models/mobile_model/face-mobile.prototxt"
        face_m = "../models/mx_models/mobile_model/face-mobile.caffemodel"
        if config.feature_1024:
            out_layer = 'fc5_n'
        elif config.insight:
            out_layer ='fc1'
        else:
            out_layer ='fc5'
        FaceModel = FaceReg(face_p,face_m,out_layer)
    else:
        print("please select a frame: caffe mxnet")
    def norm_feat(feat):
        f_mod = np.sqrt(np.sum(np.power(feat,2)))
        return feat/f_mod
    def std_norm(feat):
        mean1 = np.mean(feat)
        [lenth,] = np.shape(feat)
        f_center1 = feat-mean1
        std1 = np.sum(np.power(f_center1,2))
        std1 = np.sqrt(std1/lenth)
        norm1 = f_center1/std1
        return norm1
    if img1 is not None and img2 is not None:
        #save_mat(img1,"img1.txt")
        if config.caffe_use:
            feature_1 = FaceModel.extractfeature(img1)
            feat1 = np.array(feature_1)
            feature_2 = FaceModel.extractfeature(img2)
            feat2 = np.array(feature_2)
        elif config.mx_:
            feature_1 = FaceModel.extractfeature(img_1)
            feat1 = np.array(feature_1)
            feature_2 = FaceModel.extractfeature(img_2)
            feat2 = np.array(feature_2)
        #feat1 = np.array(feature_1)
        #np.savetxt("feat1.txt",feat1)
        if config.debug:
            print("first feat1 ",feat1[:5])
            print("first feat2 ",feat2[:5])
        #save_mat(img2,"img2.txt")
        #feature_2 = FaceModel.extractfeature(img2)
        #feat2 = np.array(feature_2)
        #np.savetxt("feat2.txt",feat2)
        if config.norm:
            feat2 = norm_feat(feat2)
            feat1 = norm_feat(feat1)
            #feat2 = std_norm(feat2)
            #feat1 = std_norm(feat1)
        d1 = L1_Sum(feat1)
        d2 = L1_Sum(feat2)
        '''
        print("L1 regularization img1,img2:",d1,d2)
        dis_s = FaceModel.calculateL2(feat1,feat2)
        dis2 = FaceModel.calculateL2(feat1,feat2,'cosine')
        dis3 = FaceModel.calculateL2(feat1,feat2,'correlation')
    cv2.imshow("img1",img_1)
    cv2.imshow("img2",img_2)
    cmd = cv2.waitKey(0) & 0xFF
    if cmd == 27 or cmd == ord('q'):
        cv2.destroyAllWindows()
    '''
    else:
        d1 = None
        d2 = None
    return d1,d2

def L1_Regular(file_in,file_out):
    '''
    file_in: saved contents, img1_path,img2_path,same_fg
    file_out: L1 regular
    '''

if __name__ == "__main__":
    parm = args()
    p1 = parm.img_path1
    p2 = parm.img_path2
    f_in = parm.file_in
    base_dir = parm.base_dir
    cmd_type = parm.cmd_type
    out_put = parm.out_file
    if cmd_type== 'imgtest':
        compare_img(p1,p2)
    elif cmd_type=='evaluate':
        compare_f(p1,p2)