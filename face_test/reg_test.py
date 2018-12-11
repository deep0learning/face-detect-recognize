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

def compare_f(p1,p2):
    f1 = np.loadtxt(p1)
    f2 = np.loadtxt(p2)
    print(f1.shape,f2.shape)
    distance = L2_distance(f1,f2,config.feature_lenth)
    distance_ = L2_distance_(f1,f2,config.feature_lenth)
    print(distance,distance_)

def compare_f_(p1,p2):
    f1 = open(p1,'r')
    f2 = open(p2,'r')
    txt1 = f1.readlines()
    txt2 = f2.readlines()
    #print(txt1)
    split1 = txt1[0].strip().split(',')
    split2 = txt2[0].strip().split(',')
    print(len(split1),len(split2))
    npa1 = map(float,split1)
    npa2 = map(float,split2)
    npa1 = np.array(npa1)
    npa2 = np.array(npa2)
    print(npa1.shape,npa2.shape)
    distance = L2_distance(npa1,npa2,512)
    distance_ = L2_distance_(npa1,npa2,512)
    print(distance,distance_)

def L1_Sum(feature):
    fe_abs = np.abs(feature)
    return np.sum(fe_abs)

def compare_img(path1,path2,model_path):
    parm = args()
    img1,img_1 = read_img(path1)
    img2,img_2 = read_img(path2)
    #model_path = "../models/tf-models/InsightFace_iter-3140"
    if config.mx_ :
        #model_path = "../models/mx_models/v1_bn/model"
        #model_path = "/home/lxy/Develop/Center_Loss/arcface/insightface/models/model-r100-ii/model"
        #model_path = "/home/lxy/Develop/Center_Loss/arcface/insightface/models/model-y1-test2/model"
        #model_path = "../models/mx_models/mobile_model/model"
        model_path = "../models/mx_models/model-100/model-org/modelresave"
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
        img_size = [112,112]
        FaceModel = TF_Face_Reg(model_path,img_size,parm.gpu)
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
        print("L1 regularization img1,img2:",d1,d2)
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
            if config.face_enhance:
                img = Img_enhance(img)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            if config.torch_:
                img = img*1.0
                img = img/255
            elif config.caffe_use:
                img = (img-127.5)*0.0078125
            feat = model_.extractfeature(img)
            if config.norm:
                #feat = norm_feat(feat)
                pass
        return feat
    feature_1 = extract_f(FaceModel,img_1)
    feat1 = np.array(feature_1)
    feature_2 = extract_f(FaceModel,img_2)
    feat2 = np.array(feature_2)
    if config.debug:
        print("second feat1 ",feat1[:5])
        print("second feat2 ",feat2[:5])
    distance_ecud = L2_distance(feat1,feat2,config.feature_lenth)
    distance_ecud2 = L2_distance_(feat1,feat2,config.feature_lenth)
    print("distance: eucli1,ecu2, %.3f  %.3f" %(distance_ecud,distance_ecud2))
    cv2.imshow("img1",img_1)
    cv2.imshow("img2",img_2)
    cmd = cv2.waitKey(0) & 0xFF
    if cmd == 27 or cmd == ord('q'):
        cv2.destroyAllWindows()


class Face_DB(object):
    def __init__(self,data):
        self.db = cKDTree(data)
    def findNeatest(self,q_data,k=1):
        d,idx = self.db.query(q_data,k)
        return d,idx

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


def Db_test(label_file,**kwargs):
    #db_file,data_file,base_dir,id_dir,dis_dir,base_id
    db_file = kwargs.get('db_file',None)
    data_file = kwargs.get('data_file',None)
    base_dir = kwargs.get('base_dir',None)
    id_dir = kwargs.get('id_dir',None)
    dis_dir = kwargs.get('saved_dir',None)
    base_id = kwargs.get('base_id',None)
    failed_dir = kwargs.get('failed_dir',None)
    top2_failed_dir = kwargs.get('top2_failed_dir',None)
    db_in = open(db_file,'r')
    data_in = open(data_file,'r')
    db_lines = db_in.readlines()
    data_lines = data_in.readlines()
    dis_record = open("./output/distance_top1.txt",'w')
    dis_top2 = open("./output/distance_top2.txt",'w')
    dis_record.write("filename,distance,l1_regular,reg_fg\n")
    dis_top2.write("filename,confidence,confidence2,l1_regular,reg_fg\n")
    if config.torch_:
        model_path = "../models/torch_models/resnet50m-xent-face1.pth.tar"
        FaceModel = Deep_Face(model_path,256,128)
    elif config.mx_:
        #model_path = "/home/lxy/Develop/Center_Loss/arcface/insightface/models/model-r50-am-lfw/model"
        #model_path = "/home/lxy/Develop/Center_Loss/arcface/insightface/models/model-r34-amf/model"
        #model_path = "../models/mx_models/v1_bn/model" #9
        #model_path = "/home/lxy/Develop/Center_Loss/arcface/insightface/models/model-r100-ii/model"
        #model_path = "/home/lxy/Develop/Center_Loss/arcface/insightface/models/model-y1-test2/model"
        #model_path = "../models/mx_models/mobile_model/model"
        #model_path = "../models/mx_models/model-100/model"
        #model_path = "../models/mx_models/model_prison/model"
        #model_path = "../models/mx_models/model-r50/model"
        #model_path = "../models/mx_models/model-100/model-v2/model" #5
        #model_path = "../models/mx_models/model-100/model-v3/model" #1
        #model_path = "../models/mx_models/model-100/model-v4/model" #11
        model_path = "../models/mx_models/model-100/model-org/modelresave"
        epoch_num = 0 #9#2
        img_size = [112,112]
        FaceModel = mx_Face(model_path,epoch_num,img_size)
    elif config.caffe_use:
        face_p1 = "../models/sphere/sph_2.prototxt"
        face_m1 = "../models/sphere/sph20_ms_4v5.caffemodel"
        face_p2 = "../models/mx_models/mobile_model/face-mobile.prototxt"
        face_m2 = "../models/mx_models/mobile_model/face-mobile.caffemodel"
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
    Face_features = ExtractFeatures(FaceModel)
    def make_dirs(dir_path):
        if os.path.exists(dir_path):
            pass
        else:
            os.makedirs(dir_path)
    make_dirs(dis_dir)
    if failed_dir is not None:
        make_dirs(failed_dir)
    if top2_failed_dir is not None:
        make_dirs(top2_failed_dir)
    parm = args()
    DB_FT = Annoy_DB(config.feature_lenth)
    tpr = 0
    fpr = 0 
    idx_ = 0
    def get_label(label_file):
        f_r = open(label_file,'rb')
        l_dict = pickle.load(f_r)
        #print(l_dict.values())
        f_r.close()
        return l_dict
    db_names = []
    save_reg_dirs = []
    db_cnt_dict = dict()
    db_paths_dict = dict()
    #db_features = []
    for item_,line_one in enumerate(db_lines):
        line_one = line_one.strip()
        base_feat = Face_features.extract_f(line_one,id_dir)
        one_name = line_one[:-4]
        db_names.append(one_name)
        face_db_dir = os.path.join(dis_dir,one_name)
        save_reg_dirs.append(face_db_dir)
        if base_feat is None:
            print("feature is None")
            continue
        DB_FT.add_data(item_,base_feat)
        db_path = os.path.join(id_dir,line_one)
        db_paths_dict[one_name] = db_path
        #db_features.append(base_feat)
    #db_features = np.asarray(db_features)
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
    test_label_dict = dict()
    reg_frame_dict = dict()
    for querry_line in data_lines:
        querry_line = querry_line.strip()
        if config.use_framenum:
            frame_num = get_frame_num(querry_line)
        if config.reg_fromlabel:
            que_spl = querry_line.split("/")
            if config.dir_label:
                key_que = que_spl[0]
            else:
                key_que = que_spl[0][2:]
            test_label_dict[key_que] = 1
            real_label = label_dict.setdefault(key_que,300)+base_id
        idx_+=1
        sys.stdout.write('\r>> deal with %d/%d' % (idx_,len(data_lines)))
        sys.stdout.flush()
        if config.face_detect:
            querry_feat = Face_features.extract_f3(querry_line,base_dir)
        else:
            querry_feat = Face_features.extract_f2(querry_line,base_dir)
        if querry_feat is None:
            print("feature is None")
            continue
        else:
            if config.getL1:
                l1_dis = L1_Sum(querry_feat)
            else:
                l1_dis = 0
            idx_list,distance_list = DB_FT.findNeatest(querry_feat,3) 
            #print("distance, idx ",distance,idx)
            idx = idx_list[0]
            img_name = db_names[idx] 
            img_dir = save_reg_dirs[idx]
            pred_label = label_dict.setdefault(img_name,300)+base_id
            img_cnt = db_cnt_dict.setdefault(img_name,0)
            org_path = os.path.join(base_dir,querry_line)
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
            #dis_record.write("%.3f\n" %(distance_list[0]))
            #dis_top2.write("%.3f\n" %(distance_list[1] - distance_list[0]))
            reg_fg = 0
            #if distance_list[0] <= config.top1_distance and (distance_list[1] - distance_list[0]) >= Threshold_Value:
            if save_condition:
                print("distance: ",distance_list[0])
                make_dirs(img_dir)
                if config.reg_fromlabel:
                    dist_path = os.path.join(img_dir,que_spl[1])
                else:
                    dist_path = os.path.join(img_dir,querry_line)
                shutil.copyfile(org_path,dist_path)  
                db_cnt_dict[img_name] = img_cnt+1 
                if db_cnt_dict[img_name] == 1 and config.save_idimg:
                    org_id_path = db_paths_dict[img_name]
                    #dist_id_path = os.path.join(img_dir,img_name+".jpg")
                    dist_id_path = os.path.join(img_dir,"front.jpg")
                    shutil.copyfile(org_id_path,dist_id_path)
                if config.reg_fromlabel:
                    print("real and pred ",real_label,pred_label,querry_line)
                    if int(pred_label) == int(real_label):
                        tpr+=1
                        reg_fg = 1
                        print("*************")
                    else:
                        fpr+=1
                        reg_fg = -1
                        if config.save_failedimg:
                            failed_img_dir = os.path.join(failed_dir,key_que)
                            make_dirs(failed_img_dir)
                            failed_img_path = os.path.join(failed_img_dir,que_spl[1])
                            shutil.copyfile(org_path,failed_img_path)
                        cnt_org = org_id_dict.setdefault(key_que,0)
                        org_id_dict[key_que] = cnt_org+1
                        cnt_dest = dest_id_dict.setdefault(img_name,0)
                        dest_id_dict[img_name] = cnt_dest+1
                        ori_dest_id = org_wrong_dict.setdefault(key_que,[])
                        if img_name in ori_dest_id:
                            pass
                        else:
                            org_wrong_dict[key_que].append(img_name)
                else:
                    print("pred label: ",pred_label,querry_line)
                    tpr+=1
                    print("*************")
            elif config.save_failedimg and not config.reg_fromlabel:
                failed_img_dir = os.path.join(failed_dir,img_name)
                make_dirs(failed_img_dir)
                failed_img_path = os.path.join(failed_img_dir,querry_line)
                shutil.copyfile(org_path,failed_img_path)
            elif config.save_top2_failedimg :
                failed_img_dir = os.path.join(top2_failed_dir,img_name)
                make_dirs(failed_img_dir)
                failed_img_path = os.path.join(failed_img_dir,que_spl[1])
                shutil.copyfile(org_path,failed_img_path)
            dis_record.write("%s,%.3f,%.3f,%d\n" %(org_path,distance_list[0],l1_dis,reg_fg))
            dis_top2.write("%s,%.3f,%.3f,%.3f,%d\n" %(org_path,distance_list[1] - distance_list[0],distance_list[2] - distance_list[0],l1_dis,reg_fg))
    right_id =0
    not_reg_id = []
    for key_name in db_cnt_dict.keys():
        if db_cnt_dict[key_name] != 0:
            right_id+=1
        elif config.reg_fromlabel :
            if key_name in test_label_dict.keys():
                not_reg_id.append(key_name)
        else:
            not_reg_id.append(key_name)
    print("not reg keys: ",not_reg_id)
    print("right id ",right_id)
    if config.reg_fromlabel:
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
        #print("values: ",db_cnt_dict.values())
        f_result.close()
        print("TPR and FPR is ",tpr,fpr)
        print("wrong orignal: ",org_id_dict.values())
        print("who will be wrong: ",dest_id_dict.values())
        print("the org and dest: ",org_id,dest_id)
    else:
        print("Reg img num: ",tpr) 
    db_in.close()
    data_in.close()
    dis_record.close()
    dis_top2.close()

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
        Db_test(label_file,db_file=db_file,data_file=data_file,\
                base_dir=base_dir,id_dir=id_dir,saved_dir=saved_dir,\
                base_id=base_id,failed_dir=failed_dir,top2_failed_dir=top2_dir)
    elif cmd_type== 'imgtest':
        compare_img(p1,p2,model_path)
    elif cmd_type=='comparefile':
        compare_f(p1,p2)
    #DB_Pred(label_file,db_file,data_file,base_dir,id_dir,saved_dir,base_id)