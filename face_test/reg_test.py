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
    parser.add_argument('--label-file', default="/output/prison_label_236.pkl",dest="label_file",\
                         type=str,help="saved labels file .pkl")
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
    elif config.caffe_use:
        FaceModel = FaceReg(0.7)
    else:
        FaceModel = TF_Face_Reg(model_path,img_size,parm.gpu)
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

class Face_DB(object):
    def __init__(self,data):
        self.db = cKDTree(data)
    def findNeatest(self,q_data,k=1):
        d,idx = self.db.query(q_data,k)
        return d,idx

class Annoy_DB(object):
    def __init__(self,data_dim):
        if config.mx_:
            self.db = AnnoyIndex(data_dim,metric='euclidean')
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

def Db_test(label_file,db_file,data_file,base_dir,id_dir,dis_dir,base_id):
    db_in = open(db_file,'r')
    data_in = open(data_file,'r')
    db_lines = db_in.readlines()
    data_lines = data_in.readlines()
    dis_record = open("./output/distance_top1.txt",'w')
    dis_top2 = open("./output/distance_top2.txt",'w')
    if config.torch_:
        model_path = "../models/torch_models/resnet50m-xent-face1.pth.tar"
        FaceModel = Deep_Face(model_path,256,128)
    elif config.mx_:
        #model_path = "/home/lxy/Develop/Center_Loss/arcface/insightface/models/model-r50-am-lfw/model"
        #model_path = "/home/lxy/Develop/Center_Loss/arcface/insightface/models/model-r34-amf/model"
        model_path = "../models/mx_models/model"
        epoch_num = 19
        img_size = [112,112]
        FaceModel = mx_Face(model_path,epoch_num,img_size)
    else:
        FaceModel = FaceReg(0.7)
    parm = args()
    img_size = [112,96]
    DB_FT = Annoy_DB(config.feature_lenth)
    tpr = 0
    fpr = 0
    #FaceModel = TF_Face_Reg(model_path,img_size,parm.gpu)
    idx_ = 0
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
        #print("feature ",feat[:5])
        return feat
    db_names = []
    db_paths = []
    db_cnt_dict = dict()
    db_features = []
    for item_,line_one in enumerate(db_lines):
        line_one = line_one.strip()
        base_feat = extract_f(FaceModel,line_one,id_dir)
        one_name = line_one[:-4]
        db_names.append(one_name)
        face_db_dir = os.path.join(dis_dir,one_name)
        db_paths.append(face_db_dir)
        if os.path.exists(face_db_dir):
            pass
        else:
            os.makedirs(face_db_dir)
        if base_feat is None:
            print("feature is None")
            continue
        DB_FT.add_data(item_,base_feat)
        db_features.append(base_feat)
    db_features = np.asarray(db_features)
    print("begin do build db")
    #print(db_features.shape)
    #DB_FT = Face_DB(db_features)
    DB_FT.build_db()
    print("db over")
    label_dict = get_label(label_file)
    dest_id_dict = dict()
    org_id_dict = dict()
    org_wrong_dict = dict()
    cnt_org = 0 
    cnt_dest = 0
    for querry_line in data_lines:
        querry_line = querry_line.strip()
        que_spl = querry_line.split("/")
        key_que = que_spl[0][2:]
        real_label = label_dict.setdefault(key_que,300)+base_id
        idx_+=1
        sys.stdout.write('\r>> deal with %d/%d' % (idx_,len(data_lines)))
        sys.stdout.flush()
        querry_feat = extract_f(FaceModel,querry_line,base_dir)
        if querry_feat is None:
            print("feature is None")
            continue
        else:
            idx_list,distance_list = DB_FT.findNeatest(querry_feat,2) 
            #print("distance, idx ",distance,idx)
            idx = idx_list[0]
            img_name = db_names[idx] 
            img_path = db_paths[idx]
            #pred_label = label_dict[img_name]
            pred_label = label_dict.setdefault(img_name,300)+base_id
            img_cnt = db_cnt_dict.setdefault(img_name,0)
            org_path = os.path.join(base_dir,querry_line)
            dist_path = os.path.join(img_path,img_name+"_"+str(img_cnt)+".jpg")
            #print(dist_path)
            print("distance ",distance_list)
            dis_record.write("%.3f\n" %(distance_list[0]))
            dis_top2.write("%.3f\n" %(distance_list[1] - distance_list[0]))
            if config.mx_:
                Threshold_Value = config.confidence
            else:
                Threshold_Value = config.confidence
            if distance_list[0] <= config.top1_distance and (distance_list[1] - distance_list[0]) >= Threshold_Value:
                print("distance: ",distance_list[0])
                shutil.copyfile(org_path,dist_path)  
                db_cnt_dict[img_name] = img_cnt+1 
                print("real and pred ",real_label,pred_label,querry_line)
                if int(pred_label) == int(real_label):
                    tpr+=1
                    print("*************")
                else:
                    fpr+=1
                    cnt_org = org_id_dict.setdefault(key_que,0)
                    org_id_dict[key_que] = cnt_org+1
                    cnt_dest = dest_id_dict.setdefault(img_name,0)
                    dest_id_dict[img_name] = cnt_dest+1
                    ori_dest_id = org_wrong_dict.setdefault(key_que,[])
                    if img_name in ori_dest_id:
                        pass
                    else:
                        org_wrong_dict[key_que].append(img_name)
    right_id =0
    not_reg_id = []
    for key_name in db_cnt_dict.keys():
        if db_cnt_dict[key_name] != 0:
            right_id+=1
        else:
            not_reg_id.append(key_name)
    org_id = 0
    #ori_name = []
    for key_name in org_id_dict.keys():
        if org_id_dict[key_name] != 0:
            org_id +=1
            #ori_name.append[key_name]
    dest_id = 0
    #dest_name = []
    for key_name in dest_id_dict.keys():
        if dest_id_dict[key_name] != 0:
            dest_id +=1
            #dest_name.append[key_name]
    f_result = open("./output/result_record.txt",'w')
    for key_name in org_wrong_dict.keys():
        f_result.write("{} : {}".format(key_name,org_wrong_dict[key_name]))
        f_result.write("\n")
    print("not reg keys: ",not_reg_id)
    #print("values: ",db_cnt_dict.values())
    print("right id ",right_id)
    print("TPR and FPR is ",tpr,fpr)
    print("wrong orignal: ",org_id_dict.values())
    print("who will be wrong: ",dest_id_dict.values())
    print("the org and dest: ",org_id,dest_id)
    db_in.close()
    data_in.close()
    f_result.close()
    dis_record.close()
    dis_top2.close()


def DB_Pred(label_file,db_file,data_file,base_dir,id_dir,dis_dir,base_id):
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


if __name__ == "__main__":
    parm = args()
    p1 = parm.img_path1
    p2 = parm.img_path2
    model_path = parm.tf_model
    base_id = parm.base_id
    #compare_img(p1,p2,model_path) 
    f_in = parm.file_in
    db_file = parm.file_in
    base_dir = parm.base_dir
    id_dir = parm.id_dir
    saved_dir = parm.save_dir
    data_file = parm.data_file
    cmd_type = parm.cmd_type
    label_file = parm.label_file
    #confuMat_test(f_in,base_dir,id_dir)
    #test()
    #feature_generate(f_in,base_dir)
    #reg_test(f_in,base_dir,id_dir)
    #label_file = "prison_label_10778.pkl"
    if cmd_type == 'dbtest':
        #label_file = "prison_label_218.pkl"
        Db_test(label_file,db_file,data_file,base_dir,id_dir,saved_dir,base_id)
    elif cmd_type== 'imgtest':
        compare_img(p1,p2,model_path)
    #label_file = "prison_label_203.pkl"
    #label_file = "prison_label_10778.pkl"
    #label_file = "prison_label_236.pkl"
    #DB_Pred(label_file,db_file,data_file,base_dir,id_dir,saved_dir,base_id)