# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/07/2 12:09
#project: Face recognize
#company: Senscape
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  caffe and tensorflow
####################################################
import sys
sys.path.append('../')
sys.path.append('/home/lxy/caffe/python')
import os
os.environ['GLOG_minloglevel'] = '2'
#import caffe
import cv2
import numpy as np
import argparse
import time
from scipy.spatial import distance
from face_config import config
#import Queue
from multiprocessing import Process,Pool,Queue
import tensorflow as tf
from face_model import FaceReg,TF_Face_Reg

def confuMat_test_async(file_in,base_dir):
    f_in = open(file_in,'r')
    f_out = open("distance.txt",'w')
    file_lines = f_in.readlines()
    FaceModel = FaceReg(0.7)
    FaceModel2 = FaceReg(0.7)
    FaceModel3 = FaceReg(0.7)
    FaceModel4 = FaceReg(0.7)
    FaceModel5 = FaceReg(0.7)
    idx = 0
    Num_ = 3
    pool = Pool(processes=Num_)
    Model_list = [FaceModel,FaceModel2,FaceModel3,FaceModel4,FaceModel5]
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
        print("begin to subpro")
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
            dis_ = 100
        else:
            dis_ = model_.calculateL2(base_feat,feat)
        print("subprocess ",dis_)
        return dis_
    def process_img_(parm_dict):
        print("begin to subpro")
        model_ = parm_dict.fmodel
        path_ = parm_dict.img_path
        base_dir = parm_dict.img_dir
        base_feat = parm_dict.feat
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
            dis_ = 100
        else:
            dis_ = model_.calculateL2(base_feat,feat)
        print("subprocess ",dis_)
        return dis_

    for line_one in file_lines:
        line_one = line_one.strip()
        tmp_splits = line_one.split(",")
        idx+=1
        sys.stdout.write('\r>> deal with %d/%d' % (idx,len(file_lines)))
        sys.stdout.flush()
        base_feat = extract_f(FaceModel,tmp_splits[0],base_dir)
        if base_feat is None:
            continue
        cnt_process = 0
        img_path_list = []
        cnt_idx = 0
        splt_len = len(tmp_splits)
        for i in range(1,splt_len):
            '''
            tmp_feat = extract_f(FaceModel,tmp_splits[i],base_dir)
            if tmp_feat is None:
                dis_ = 100
            else:
                dis_ = FaceModel.calculateL2(base_feat,tmp_feat)
            '''
            img_path_list.append(tmp_splits[i])
            cnt_process+=1
            cnt_idx+=1
            print("cnt_process ",cnt_process)
            #dis = process_img_(FaceModel,img_path_list[i-1],base_dir,base_feat)
            if (splt_len < Num_ and cnt_process == splt_len) or (splt_len %Num_ and cnt_idx == splt_len):
                multi_pro = []
                cnt_process = 0
                for jk in range(len(img_path_list)):
                    res = pool.apply_async(process_img_,args=(FaceModel,img_path_list[jk],base_dir,base_feat,))
                    #res = pool.apply_async(test_process_, args=(i,0,))
                    multi_pro.append(res)
                pool.close()
                pool.join()
                for res in multi_pro:
                    tmp_rt = res.get()
                    f_out.write("{} ".format(tmp_rt))
                img_path_list = []
            if cnt_process ==Num_:
                print("begin to child pro")
                multi_pro = []
                cnt_process = 0
                parm_list = []
                for jk in range(Num_):
                    parm_dit = dict()
                    parm_dit["fmodel"] = Model_list[jk]
                    parm_dit["img_path"] = img_path_list[jk]
                    parm_dit["img_dir"] = base_dir
                    parm_dit["feat"] = base_feat
                    parm_list.append(parm_dit)
                res = pool.map(process_img_,parm_list)
                '''
                for jk in range(Num_):
                    res = pool.apply_async(func=process_img_,args=(jk,img_path_list[jk],base_dir,base_feat,))
                    #res = pool.apply_async(process_img_, args=(FaceModel,0,))
                    multi_pro.append(res)
                '''
                pool.close()
                pool.join()
                print("all subprocess done",len(multi_pro))
                '''
                for res in multi_pro:
                    #print(res.get())
                    tmp_rt = res.get()
                    f_out.write("{} ".format(tmp_rt))
                '''
                img_path_list = []
        f_out.write("\n")
    f_out.close()
    f_in.close()

def test_process(name,q_):
    print ('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(3)
    q_.put(name)
    end = time.time()
    print ('Task %s runs %0.2f seconds.' % (name, (end - start)))
def test_process_(name,k):
    print ('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(3)
    end = time.time()
    print ('Task %s runs %0.2f seconds.' % (name, (end - start)))
    return [name,k]
def test():
    print ('Parent process %s.' % os.getpid())
    p = Pool()
    '''
    q_f = Queue()
    p1 = Process(target=test_process,args=(0,q_f,))
    p2 = Process(target=test_process,args=(1,q_f,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    '''
    mul = []
    for i in range(5):
        res = p.apply_async(test_process_, args=(i,0,))
        mul.append(res)
    
    #res = p.map(test_process_,[0,1,2,3],[0,0,0,0])
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    #print(q_f.get())
    #print(q_f.get())
    #print(res)
    for res in mul:
        print(res.get())