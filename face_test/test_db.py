# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/12/26 14:09
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
    parser.add_argument('--data-file',type=str,dest='data_file',default='None',\
                        help="the file input path")
    parser.add_argument('--min-size',type=int,dest='min_size',default=50,\
                        help="scale img size")
    parser.add_argument('--img-path1',type=str,dest='img_path1',default="test1.jpg",\
                        help="img1 saved path")
    parser.add_argument('--img-path2',type=str,dest='img_path2',default="test2.jpg",\
                        help="scale img size")
    parser.add_argument('--base-dir',type=str,dest='base_dir',default="./base_dir",\
                        help="images saved dir")
    parser.add_argument('--prototxt',type=str,dest='p_path',default="../models/deploy.prototxt",\
                        help="caffe prototxt path")
    parser.add_argument('--caffemodel',type=str,dest='m_path',default="../models/deploy.caffemodel",\
                        help="caffe model path")
    parser.add_argument('--id-dir',type=str,dest='id_dir',default="./id_dir",\
                        help="images saved dir")
    parser.add_argument('--tf-model',type=str,dest='tf_model',default="../models/",\
                        help="models saved dir")
    parser.add_argument('--gpu', default='0', type=str,help='which gpu to run')
    parser.add_argument('--save-dir',type=str,dest='save_dir',default="./saved_dir",\
                        help="images saved dir")
    parser.add_argument('--failed-dir',type=str,dest='failed_dir',default="./failed_dir",\
                        help="fpr saved dir")
    parser.add_argument('--top2-failed-dir',type=str,dest='top2_failed_dir',default="./top2_failed_dir",\
                        help="fpr saved dir")
    parser.add_argument('--base-id', default=0,dest='base_id', type=int,help='label plus the id')
    parser.add_argument('--cmd-type', default="dbtest",dest='cmd_type', type=str,\
                        help="which code to run: dbtest,imgtest ")
    parser.add_argument('--label-file', default="/output/prison_label_236.pkl",dest="label_file",\
                         type=str,help="saved labels file .pkl")
    return parser.parse_args()

class Annoy_DB(object):
    def __init__(self,data_dim):
        if config.mx_:
            #self.db = AnnoyIndex(data_dim,metric='euclidean')
            self.db = AnnoyIndex(data_dim,metric='angular')
        else:
            self.db = AnnoyIndex(data_dim,metric='angular')
    def add_data(self,i,data):
        self.db.add_item(i,data)
    def build_db(self,tree_num=64):
        self.db.build(tree_num)
    def findNeatest(self,q_data,k=1):
        d,idx = self.db.get_nns_by_vector(q_data,k,include_distances=True)
        #return d[0],idx[0]
        return d,idx
   
def Img_enhance(image,fgamma=1.5):
    image_gamma = np.uint8(np.power((np.array(image)/255.0),fgamma)*255.0)
    cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
    cv2.convertScaleAbs(image_gamma, image_gamma)
    return image_gamma

def get_frame_num(img_path):
    '''
    img_path: the image name include frame_num
    '''
    que_spl = img_path.split("/")
    face_num = que_spl[1].split("_")
    face_num = face_num[1][:-4]
    return int(face_num)


class DB_Test(object):
    def __init__(self,label_file,**kwargs):
        #db_file,data_file,base_dir,id_dir,dis_dir,base_id
        self.db_file = kwargs.get('db_file',None)
        self.data_file = kwargs.get('data_file',None)
        self.base_dir = kwargs.get('base_dir',None)
        self.id_dir = kwargs.get('id_dir',None)
        self.dis_dir = kwargs.get('saved_dir',None)
        self.base_id = kwargs.get('base_id',None)
        self.failed_dir = kwargs.get('failed_dir',None)
        self.top2_failed_dir = kwargs.get('top2_failed_dir',None)
        self.open_files()
        self.load_model()
        self.build_dir()
        self.DB_FT = Annoy_DB(config.feature_lenth)
        self.label_dict = self.get_label(label_file)
        self.build_database()

    def open_files(self):
        db_in = open(self.db_file,'r')
        data_in = open(self.data_file,'r')
        self.db_lines = db_in.readlines()
        self.data_lines = data_in.readlines()
        self.test_data_num = len(self.data_lines)
        #dis_record = open("./output/distance_top1.txt",'w')
        self.dis_top2 = open("./output/distance_top2.txt",'w')
        #dis_record.write("filename,distance,l1_regular,reg_fg\n")
        self.dis_top2.write("filename,distance,confidence,confidence2,l1_regular,reg_fg,real_label,pred_label\n")
        db_in.close()
        data_in.close()

    def load_model(self):
        if config.mx_:
            model_path = "../models/mx_models/model-100/model-org/modelresave"
            epoch_num = 0 #9#2
            img_size = [112,112]
            FaceModel = mx_Face(model_path,epoch_num,img_size)
        elif config.caffe_use:
            #face_p2 = "../models/mx_models/mobile_model/face-mobile.prototxt"
            #face_m2 = "../models/mx_models/mobile_model/face-mobile.caffemodel"
            face_p2 = "../models/mx_models/face-100.prototxt"
            face_m2 = "../models/mx_models/face-100.caffemodel"
            if config.feature_1024:
                out_layer = 'fc5_n'
            elif config.insight:
                out_layer ='fc1'
            else:
                out_layer ='fc5'
            FaceModel = FaceReg(face_p2,face_m2,out_layer)
        else:
            print("please select model frame")
        #mkdir save image dir
        self.Face_features = ExtractFeatures(FaceModel)

    def make_dirs(self,dir_path):
        if os.path.exists(dir_path):
            pass
        else:
            os.makedirs(dir_path)

    def build_dir(self):
        self.make_dirs(self.dis_dir)
        if self.failed_dir is not None:
            self.make_dirs(self.failed_dir)
        if self.top2_failed_dir is not None:
            self.make_dirs(self.top2_failed_dir)

    
    def get_label(self,label_file):
        f_r = open(label_file,'rb')
        l_dict = pickle.load(f_r)
        #print(l_dict.values())
        f_r.close()
        return l_dict

    def build_database(self):
        self.db_names = []
        self.save_reg_dirs = []
        self.db_paths_dict = dict()
        #db_features = []
        t = time.time()
        for item_,line_one in enumerate(self.db_lines):
            sys.stdout.write('\r>> deal with %d/%d' % (item_,len(self.db_lines)))
            line_one = line_one.strip()
            base_feat = self.Face_features.extract_f(line_one,self.id_dir)
            one_name = line_one[:-4]
            self.db_names.append(one_name)
            face_db_dir = os.path.join(self.dis_dir,one_name)
            self.save_reg_dirs.append(face_db_dir)
            if base_feat is None:
                print("feature is None")
                continue
            self.DB_FT.add_data(item_,base_feat)
            db_path = os.path.join(self.id_dir,line_one)
            self.db_paths_dict[one_name] = db_path
        print("begin do build db")
        self.DB_FT.build_db()
        print("db over")
        self.build_db_time = time.time()-t

    def L1_Sum(self,feature):
        fe_abs = np.abs(feature)
        return np.sum(fe_abs)

    def callback(self):
        self.tpr = 0
        self.fpr = 0 
        idx_ = 0
        self.db_cnt_dict = dict()
        self.dest_id_dict = dict()
        self.org_id_dict = dict()
        self.org_wrong_dict = dict()
        cnt_org = 0 
        cnt_dest = 0
        self.test_label_dict = dict()
        reg_frame_dict = dict()
        for querry_line in self.data_lines:
            querry_line = querry_line.strip()
            if config.use_framenum:
                frame_num = get_frame_num(querry_line)
            if config.reg_fromlabel:
                if config.highway:
                    if '-' in querry_line.strip():
                        que_spl = querry_line.split('-')
                    elif '_' in querry_line.strip():
                        que_spl = querry_line.split('_')
                    else:
                        print('not known mark:',querry_line)
                    key_que = que_spl[0]
                else:
                    que_spl = querry_line.split("/")
                    if config.dir_label:
                        key_que = que_spl[0]
                    else:
                        key_que = que_spl[0][2:]
                self.test_label_dict[key_que] = 1
                real_label = self.label_dict.setdefault(key_que,300)+self.base_id
            idx_+=1
            sys.stdout.write('\r>> deal with %d/%d' % (idx_,self.test_data_num))
            sys.stdout.flush()
            if config.face_detect:
                querry_feat = self.Face_features.extract_f3(querry_line,self.base_dir)
            else:
                querry_feat = self.Face_features.extract_f2(querry_line,self.base_dir)
            if querry_feat is None:
                print("feature is None")
                continue
            else:
                if config.getL1:
                    l1_dis = self.L1_Sum(querry_feat)
                else:
                    l1_dis = 0
                dbtime1 = time.time()
                idx_list,distance_list = self.DB_FT.findNeatest(querry_feat,3)
                dbtime2 = time.time()-dbtime1
                if idx_ > self.test_data_num - 100 :
                    print("look up table time: ",dbtime2)
                idx = idx_list[0]
                img_name = self.db_names[idx] 
                img_dir = self.save_reg_dirs[idx]
                pred_label = self.label_dict.setdefault(img_name,300)+self.base_id
                img_cnt = self.db_cnt_dict.setdefault(img_name,0)
                org_path = os.path.join(self.base_dir,querry_line)
                Threshold_Value = config.confidence
                reg_condition = 1 if(distance_list[0] <= config.top1_distance and (distance_list[1] - distance_list[0]) >= Threshold_Value) else 0
                if config.use_framenum :
                    if reg_condition:
                        reg_frame_cnt = reg_frame_dict.setdefault(img_name,frame_num)
                        frame_interval = np.abs(frame_num - reg_frame_cnt)
                        save_condition = 1 if frame_interval < config.frame_interval else 0
                    else:
                        save_condition = 0
                else:
                    save_condition = reg_condition
                print("distance ",distance_list)
                reg_fg = 0
                if save_condition:
                    print("distance: ",distance_list[0])
                    self.make_dirs(img_dir)
                    if config.reg_fromlabel:
                        if config.highway:
                            dist_path = os.path.join(img_dir,querry_line)
                        else:
                            dist_path = os.path.join(img_dir,que_spl[1])
                    else:
                        dist_path = os.path.join(img_dir,querry_line)
                    shutil.copyfile(org_path,dist_path)  
                    self.db_cnt_dict[img_name] = img_cnt+1 
                    if self.db_cnt_dict[img_name] == 1 and config.save_idimg:
                        org_id_path = self.db_paths_dict[img_name]
                        dist_id_path = os.path.join(img_dir,img_name+".jpg")
                        #dist_id_path = os.path.join(img_dir,"front.jpg")
                        shutil.copyfile(org_id_path,dist_id_path)
                    if config.reg_fromlabel:
                        print("real and pred ",real_label,pred_label,querry_line)
                        if int(pred_label) == int(real_label):
                            self.tpr+=1
                            reg_fg = 1
                            print("*************")
                        else:
                            self.fpr+=1
                            reg_fg = -1
                            if config.save_failedimg:
                                failed_img_dir = os.path.join(self.failed_dir,key_que)
                                self.make_dirs(failed_img_dir)
                                if config.highway:
                                    failed_img_path = os.path.join(failed_img_dir,querry_line)
                                else:
                                    failed_img_path = os.path.join(failed_img_dir,que_spl[1])
                                shutil.copyfile(org_path,failed_img_path)
                            cnt_org = self.org_id_dict.setdefault(key_que,0)
                            self.org_id_dict[key_que] = cnt_org+1
                            cnt_dest = self.dest_id_dict.setdefault(img_name,0)
                            self.dest_id_dict[img_name] = cnt_dest+1
                            ori_dest_id = self.org_wrong_dict.setdefault(key_que,[])
                            if img_name in ori_dest_id:
                                pass
                            else:
                                self.org_wrong_dict[key_que].append(img_name)
                    else:
                        print("pred label: ",pred_label,querry_line)
                        self.tpr+=1
                        print("*************")
                elif config.save_failedimg and not config.reg_fromlabel:
                    failed_img_dir = os.path.join(self.failed_dir,img_name)
                    self.make_dirs(failed_img_dir)
                    failed_img_path = os.path.join(failed_img_dir,querry_line)
                    shutil.copyfile(org_path,failed_img_path)
                elif config.save_top2_failedimg :
                    failed_img_dir = os.path.join(self.top2_failed_dir,img_name)
                    self.make_dirs(failed_img_dir)
                    if config.highway:
                        failed_img_path = os.path.join(failed_img_dir,querry_line)
                    else:
                        failed_img_path = os.path.join(failed_img_dir,que_spl[1])
                    shutil.copyfile(org_path,failed_img_path)
                #self.dis_record.write("%s,%.3f,%.3f,%d\n" %(org_path,distance_list[0],l1_dis,reg_fg))
                self.dis_top2.write("%s,%.3f,%.3f,%.3f,%.3f,%d,%s,%s\n" %(org_path,distance_list[0], \
                        distance_list[1] - distance_list[0],distance_list[2] - distance_list[0],l1_dis,reg_fg,real_label,pred_label))
        #self.dis_record.close()
        self.dis_top2.close()
    
    def write_record(self):
        right_id =0
        not_reg_id = []
        for key_name in self.db_cnt_dict.keys():
            if self.db_cnt_dict[key_name] != 0:
                right_id+=1
            elif config.reg_fromlabel :
                if key_name in self.test_label_dict.keys():
                    not_reg_id.append(key_name)
            else:
                not_reg_id.append(key_name)
        print("not reg keys: ",not_reg_id)
        print("right id ",right_id)
        if config.reg_fromlabel:
            org_id = 0
            #ori_name = []
            for key_name in self.org_id_dict.keys():
                if self.org_id_dict[key_name] != 0:
                    org_id +=1
                    #ori_name.append[key_name]
            dest_id = 0
            #dest_name = []
            for key_name in self.dest_id_dict.keys():
                if self.dest_id_dict[key_name] != 0:
                    dest_id +=1
                    #dest_name.append[key_name]
            f_result = open("./output/result_record.txt",'w')
            f_result.write(str(self.build_db_time)+'\n')
            for key_name in self.org_wrong_dict.keys():
                f_result.write("{} : {}".format(key_name,self.org_wrong_dict[key_name]))
                f_result.write("\n")
            #print("values: ",db_cnt_dict.values())
            f_result.close()
            print("TPR and FPR is ",self.tpr,self.fpr)
            #print("wrong orignal: ",self.org_id_dict.values())
            #print("who will be wrong: ",self.dest_id_dict.values())
            print("the org and dest: ",org_id,dest_id)
        else:
            print("Reg img num: ",self.tpr) 


if __name__ == "__main__":
    parm = args()
    p1 = parm.img_path1
    p2 = parm.img_path2
    model_path = parm.tf_model
    base_id = parm.base_id 
    f_in = parm.file_in
    db_file = parm.file_in
    base_dir = parm.base_dir
    id_dir = parm.id_dir
    saved_dir = parm.save_dir
    data_file = parm.data_file
    cmd_type = parm.cmd_type
    label_file = parm.label_file
    failed_dir = parm.failed_dir
    top2_dir = parm.top2_failed_dir
    #confuMat_test(f_in,base_dir,id_dir)
    #test()
    #feature_generate(f_in,base_dir)
    #reg_test(f_in,base_dir,id_dir)
    if cmd_type == 'dbtest':
        #label_file = "prison_label_218.pkl"
        DB = DB_Test(label_file,db_file=db_file,data_file=data_file,\
                base_dir=base_dir,id_dir=id_dir,saved_dir=saved_dir,\
                base_id=base_id,failed_dir=failed_dir,top2_failed_dir=top2_dir)
        DB.callback()
        DB.write_record()