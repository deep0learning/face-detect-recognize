# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/06/14 14:09
#project: Face recognize
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  test caffe model and figure confusionMatrix
####################################################
import sys
sys.path.append('../')
import os
os.environ['GLOG_minloglevel'] = '2'
import cv2
import numpy as np
import argparse
import time
from scipy.spatial import distance,cKDTree
from face_config import config
from face_model import mx_Face,L2_distance
import h5py
import shutil
from annoy import AnnoyIndex
import pickle


def args():
    parser = argparse.ArgumentParser(description="mtcnn caffe")
    parser.add_argument('--file-in',type=str,dest='file_in',default='None',\
                        help="the file input path")
    parser.add_argument('--data-file',type=str,dest='data_file',default='None',\
                        help="the file input path")
    parser.add_argument('--min-size',type=int,dest='min_size',default=50,\
                        help="scale img size")
    parser.add_argument('--img-path1',type=str,dest='img_path1',default="test1.jpg",\
                        help="img1 saved path")
    parser.add_argument('--img-path2',type=str,dest='img_path2',default="test2.jpg",\
                        help="scale img size")
    parser.add_argument('--base-dir',type=str,dest='base_dir',default="./",\
                        help="images saved dir")
    parser.add_argument('--prototxt',type=str,dest='p_path',default="../models/deploy.prototxt",\
                        help="caffe prototxt path")
    parser.add_argument('--caffemodel',type=str,dest='m_path',default="../models/deploy.caffemodel",\
                        help="caffe model path")
    parser.add_argument('--id-dir',type=str,dest='id_dir',default="./",\
                        help="images saved dir")
    parser.add_argument('--tf-model',type=str,dest='tf_model',default="../models/",\
                        help="models saved dir")
    parser.add_argument('--gpu', default='0', type=str,help='which gpu to run')
    parser.add_argument('--save-dir',type=str,dest='save_dir',default="./",\
                        help="images saved dir")
    parser.add_argument('--base-id', default=0,dest='base_id', type=int,help='label plus the id')
    parser.add_argument('--cmd-type', default="dbtest",dest='cmd_type', type=str,\
                        help="which code to run: dbtest,imgtest ")
    parser.add_argument('--label-file', default="/output/",dest="label_file",\
                         type=str,help="saved labels file .pkl")
    return parser.parse_args()


def compare_img(path1,path2,model_path):
    parm = args()
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img_1 = img1
    img_2 = img2
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
    img1 = (img1-127.5)*0.0078125
    img2 = (img2-127.5)*0.0078125
    #model_path = "../models/tf-models/InsightFace_iter-3140"
    img_size = [112,112]
    if config.mx_ :
        model_path = "../models/mx_models/model"
        epoch_num = 6
        img_size = [112,112]
        FaceModel = mx_Face(model_path,epoch_num,img_size)
    if img1 is not None and img2 is not None:
        feature_1 = FaceModel.extractfeature(img1)
        feat1 = np.array(feature_1)
        if config.debug:
            print("first feat1 ",feature_1[:5])
        feature_2 = FaceModel.extractfeature(img2)
        feat2 = np.array(feature_2)
        if config.debug:
            print("first feat2 ",feature_2[:5])
        dis_s = FaceModel.calculateL2(feat1,feat2)
        dis2 = FaceModel.calculateL2(feat1,feat2,'cosine')
        dis3 = FaceModel.calculateL2(feat1,feat2,'correlation')
    if config.debug:
        print("feat1 ",feat1[:5])
        print("feat2 ",feat2[:5])
    print("distance: eucli, cos, cor %.3f,%.3f,%.3f" %(dis_s,dis2,dis3))
    def extract_f(model_,img):
        #img_path = os.path.join(base_dir,path_)
        #img = cv2.imread(img_path)
        if img is None:
            feat = None
            print("read img faild, " + img_path)
        else:
            img = Img_enhance(img)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            if config.torch_:
                img = img*1.0
                img = img/255
            else:
                img = (img-127.5)*0.0078125
            feat = model_.extractfeature(img)
            if config.norm:
                fea_std = np.std(feat)
                fea_mean = np.mean(feat)
                feat = (feat-fea_mean)/fea_std
        return feat
    feature_1 = extract_f(FaceModel,img_1)
    feat1 = np.array(feature_1)
    feature_2 = extract_f(FaceModel,img_2)
    feat2 = np.array(feature_2)
    if config.debug:
        print("feat1 ",feat1[:5])
        print("feat2 ",feat2[:5])
    distance_ecud = L2_distance(feat1,feat2,512)
    print("distance: eucli, %.3f" %(distance_ecud))
    cv2.imshow("img1",img_1)
    cv2.imshow("img2",img_2)
    cmd = cv2.waitKey(0) & 0xFF
    if cmd == 27 or cmd == ord('q'):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parm = args()
    p1 = parm.img_path1
    p2 = parm.img_path2
    model_path = parm.tf_model
    compare_img(p1,p2,model_path)
