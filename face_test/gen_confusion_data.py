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
from face_model import FaceReg,TF_Face_Reg,Deep_Face,mx_Face,L2_distance
import h5py
import shutil
from annoy import AnnoyIndex
import pickle


def confuMat_test(file_in,base_dir,id_dir):
    f_in = open(file_in,'r')
    f_out = open("distance_eculidence2.txt",'w')
    f_out2 = open("distance_cosin2.txt",'w')
    f_out3 = open("distance_correlation2.txt",'w')
    f_out4 = open("distance_canberra2.txt",'w')
    f_out5 = open("distance_bcd2.txt",'w')
    f_out6 = open("distance_chebyshev2.txt",'w')
    file_lines = f_in.readlines()
    img_size = [112,96]
    parm = args()
    model_path = parm.tf_model
    if config.caffe_use:
        FaceModel = FaceReg(0.7)
    else:
        FaceModel = TF_Face_Reg(model_path,img_size,parm.gpu)
    idx = 0
    def extract_f(model_,path_,base_dir):
        img_path = os.path.join(base_dir,path_)
        img = cv2.imread(img_path)
        if img is None:
            feat = None
            print("read img faild, " + img_path)
        else:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = (img-127.5)*0.0078125
            feat = model_.extractfeature(img)
        return feat
    def process_img(model_,path_,base_dir,base_feat):
        #print("begin to subpro")
        img_path = os.path.join(base_dir,path_)
        img = cv2.imread(img_path)
        if img is None:
            feat = None
            print("read img faild, " + img_path)
        else:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = (img-127.5)*0.0078125
            feat = model_.extractfeature(img)
        if feat is None:
            print("feat is None")
            dis_ = 100
        else:
            dis_ = model_.calculateL2(base_feat,feat)
            dis2 = model_.calculateL2(base_feat,feat,"cosine")
            dis3 = model_.calculateL2(base_feat,feat,"correlation")
            dis4 = model_.calculateL2(base_feat,feat,'canberra')
            dis5 = model_.calculateL2(base_feat,feat,'braycurtis')
            dis6 = model_.calculateL2(base_feat,feat,'chebyshev')
        #print("subprocess ",dis_)
        return [dis_,dis2,dis3,dis4,dis5,dis6]
    for line_one in file_lines:
        line_one = line_one.strip()
        tmp_splits = line_one.split(",")
        idx+=1
        sys.stdout.write('\r>> deal with %d/%d' % (idx,len(file_lines)))
        sys.stdout.flush()
        base_feat = extract_f(FaceModel,tmp_splits[0],id_dir)
        if base_feat is None:
            continue
        splt_len = len(tmp_splits)
        for i in range(1,splt_len):
            #tmp_feat = extract_f(FaceModel,tmp_splits[i],base_dir) 
            tmp_dis = process_img(FaceModel,tmp_splits[i],base_dir,base_feat)       
            f_out.write("{} ".format(tmp_dis[0]))
            f_out2.write("{} ".format(tmp_dis[1]))
            f_out3.write("{} ".format(tmp_dis[2]))
            f_out4.write("{} ".format(tmp_dis[3]))
            f_out5.write("{} ".format(tmp_dis[4]))
            f_out6.write("{} ".format(tmp_dis[5]))
        f_out.write("\n")
        f_out2.write("\n")
        f_out3.write("\n")
        f_out4.write("\n")
        f_out5.write("\n")
        f_out6.write("\n")
    f_out.close()
    f_out2.close()
    f_out3.close()
    f_out4.close()
    f_out5.close()
    f_out6.close()
    f_in.close()



def feature_generate(file_in,base_dir):
    f_in = open(file_in,'r')
    file_lines = f_in.readlines()
    FaceModel = FaceReg(0.7)
    h5_filename = "feature.h5"
    idx = 0
    def extract_f(model_,path_,base_dir):
        img_path = os.path.join(base_dir,path_)
        img = cv2.imread(img_path)
        if img is None:
            feat = None
            print("read img faild, " + img_path)
        else:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = (img-127.5)*0.0078125
            feat = model_.extractfeature(img)
        return feat
    for line_one in file_lines:
        line_one = line_one.strip()
        tmp_splits = line_one.split(",")
        splt_len = len(tmp_splits)
        feature_list = []
        label_list = []
        for i in range(1,splt_len):
            idx+=1
            sys.stdout.write('\r>> deal with %d/%d' % (idx,len(file_lines)))
            sys.stdout.flush()
            line_spl = tmp_splits[i].split("/")
            tmp_feat = extract_f(FaceModel,tmp_splits[i],base_dir)  
            feature_list.append(tmp_feat)
            label_list.append(int(line_spl[0]))
            print("lable ",line_spl[0])
        write_hdf5(h5_filename,feature_list,label_list)
        break    
    f_in.close()


def reg_test(file_in,base_dir,id_dir):
    f_in = open(file_in,'r')
    f_out = open("result.txt",'w')
    file_lines = f_in.readlines()
    FaceModel = FaceReg(0.7)
    idx = 0
    cv2.namedWindow("db")
    cv2.namedWindow("probs")
    cv2.moveWindow("db",1400,10)
    cv2.moveWindow("probs",1110,10)
    def extract_f(model_,path_,base_dir):
        img_path = os.path.join(base_dir,path_)
        img = cv2.imread(img_path)
        img_s = cv2.resize(img,(96,112))
        cv2.imshow("db",img_s)
        cv2.waitKey(1000)
        if img is None:
            feat = None
            print("read img faild, " + img_path)
        else:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = (img-127.5)*0.0078125
            feat = model_.extractfeature(img)
        return feat
    def process_img(model_,path_,base_dir,base_feat):
        #print("begin to subpro")
        img_path = os.path.join(base_dir,path_)
        img = cv2.imread(img_path)
        img_org = img
        if img is None:
            feat = None
            print("read img faild, " + img_path)
        else:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = (img-127.5)*0.0078125
            feat = model_.extractfeature(img)
        if feat is None:
            dis_ = 100
        else:
            dis_ = model_.calculateL2(base_feat,feat)
            dis2 = model_.calculateL2(base_feat,feat,"cosine")
            dis3 = model_.calculateL2(base_feat,feat,"correlation")
            dis4 = model_.calculateL2(base_feat,feat,'canberra')
            dis5 = model_.calculateL2(base_feat,feat,'braycurtis')
            dis6 = model_.calculateL2(base_feat,feat,'chebyshev')
        #print("subprocess ",dis_)
        return [dis_,dis2,dis3,dis4,dis5,dis6],img_org
    for line_one in file_lines:
        line_one = line_one.strip()
        tmp_splits = line_one.split(",")
        idx+=1
        base_feat = extract_f(FaceModel,tmp_splits[0],id_dir)
        if base_feat is None:
            print("id faild ",tmp_splits[0])
            continue
        splt_len = len(tmp_splits)
        print("***************id***********",tmp_splits[0])
        f_out.write("{}: ".format(tmp_splits[0]))
        for i in range(1,splt_len):
            #tmp_feat = extract_f(FaceModel,tmp_splits[i],base_dir) 
            tmp_dis, img_show = process_img(FaceModel,tmp_splits[i],base_dir,base_feat)  
            if tmp_dis[1] < 0.7:
                print("the corresponding: %s,  %.3f" %(tmp_splits[i], tmp_dis[1]) ) 
                img_show = cv2.resize(img_show,(96,112))
                cv2.imshow("probs",img_show) 
                cv2.waitKey(1000)  
                f_out.write("{} ".format(tmp_splits[i]))
        f_out.write("\n")
    f_out.close()
    f_in.close()

def DB_Pred(label_file,db_file,data_file,base_dir,id_dir,save_dir,base_id):
    db_in = open(db_file,'r')
    data_in = open(data_file,'r')
    db_lines = db_in.readlines()
    data_lines = data_in.readlines()
    FaceModel = FaceReg(0.7)
    parm = args()
    img_size = [112,96]
    #FaceModel = TF_Face_Reg(model_path,img_size,parm.gpu)
    idx_ = 0
    total_right = 0
    def get_label(label_file):
        f_r = open(label_file,'rb')
        l_dict = pickle.load(f_r)
        #print(l_dict.values())
        f_r.close()
        return l_dict
    def extract_f(model_,path_,base_dir):
        img_path = os.path.join(base_dir,path_)
        img = cv2.imread(img_path)
        if img is None:
            feat = None
            print("read img faild, " + img_path)
        else:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = (img-127.5)*0.0078125
            feat = model_.prob_cls(img)
        return feat
    db_names = []
    db_paths_dict = dict()
    db_cnt_dict = dict()
    db_features = []
    for item_,line_one in enumerate(db_lines):
        line_one = line_one.strip()
        one_name = line_one[:-4]
        db_names.append(one_name)
        face_db_dir = os.path.join(dis_dir,one_name)
        db_paths_dict[one_name] = face_db_dir
        if os.path.exists(face_db_dir):
            pass
        else:
            os.makedirs(face_db_dir)
    label_dict = get_label(label_file)
    for querry_line in data_lines:
        querry_line = querry_line.strip()
        que_spl = querry_line.split("/")
        key_que = que_spl[0][2:]
        #real_label = label_dict[key_que]
        real_label = label_dict.setdefault(key_que,300)+base_id
        idx_+=1
        sys.stdout.write('\r>> deal with %d/%d' % (idx_,len(data_lines)))
        sys.stdout.flush()
        querry_feat = extract_f(FaceModel,querry_line,base_dir)
        if querry_feat is None:
            print("feature is None")
            continue
        else:
            pred_label = np.argmax(querry_feat) 
            #print("distance, idx ",distance,idx)
            img_name = key_que 
            img_path = db_paths_dict[key_que]
            img_cnt = db_cnt_dict.setdefault(img_name,0)
            org_path = os.path.join(base_dir,querry_line)
            dist_path = os.path.join(img_path,img_name+"_"+str(img_cnt)+".jpg")
            #print(dist_path)
            if int(pred_label) == int(real_label):
                total_right+=1
                shutil.copyfile(org_path,dist_path)  
                db_cnt_dict[img_name] = img_cnt+1 
            else:
                print("img path: ",querry_line)
                print("real and pred: ",(real_label,pred_label))
    right_id =0
    for key_name in db_cnt_dict.keys():
        if db_cnt_dict[key_name] != 0:
            right_id+=1
    print("keys: ",db_cnt_dict.keys())
    print("values: ",db_cnt_dict.values())
    print("all right ",total_right,len(data_lines))
    print("right id ",right_id)
    db_in.close()
    data_in.close()