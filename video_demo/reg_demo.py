# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/07/24 14:09
#project: Face reg
#company: Senscape
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face detect and face reg
####################################################
import sys
sys.path.append('/home/lxy/caffe/python')
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
import cv2
import numpy as np
import argparse
import time
from video_config import config as videoconfig
sys.path.append(os.path.join(os.path.dirname(__file__),'..', 'demo'))
from mtcnn_config import config as detconfig
from Detector import FaceDetector_Opencv,MTCNNDet
import shutil
from align import Align_img , alignImg
from scipy.spatial import distance,cKDTree
sys.path.append(os.path.join(os.path.dirname(__file__),'..', 'face_test'))
from face_config import config as faceconfig
from face_model import FaceReg,TF_Face_Reg,Deep_Face,mx_Face
import h5py
from annoy import AnnoyIndex
import pickle
import logging
#import struct

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def args():
    parser = argparse.ArgumentParser(description="mtcnn caffe")
    parser.add_argument('--file-in',type=str,dest='file_in',default='None',\
                        help="the file input path")
    parser.add_argument('--id-file',type=str,dest='id_file',default='None',\
                        help="the file input path")
    parser.add_argument('--min-size',type=int,dest='min_size',default=50,\
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
    parser.add_argument('--frame-num', default=None,dest='frame_num', type=float,help='begin to run num')
    parser.add_argument('--out-file', default=None,dest='out_file', type=str,help='save output')
    return parser.parse_args()

class Annoy_DB(object):
    def __init__(self,data_dim):
        if faceconfig.mx_:
            self.db = AnnoyIndex(data_dim,metric='angular')
        else:
            self.db = AnnoyIndex(data_dim,metric='angular')
    def add_data(self,i,data):
        self.db.add_item(i,data)
    def build_db(self,tree_num=64):
        self.db.build(tree_num)
    def findNearest(self,q_data,k=1):
        d,idx = self.db.get_nns_by_vector(q_data,k,include_distances=True)
        #return d[0],idx[0]
        return d,idx
   
class DB_Reg(object):
    def __init__(self,db_file,id_dir,save_dir,label_file=None):
        if faceconfig.torch_:
            model_path = "../models/torch_models/resnet50m-xent-face1.pth.tar"
            FaceModel = Deep_Face(model_path,256,128)
        elif faceconfig.mx_:
            model_path = "/home/lxy/Develop/Center_Loss/arcface/insightface/models/model-r100-ii/model"
            #model_path = "/home/lxy/Develop/Center_Loss/arcface/insightface/models/model-r50-am-lfw/model"
            #model_path = "/home/lxy/Develop/Center_Loss/arcface/insightface/models/model-r34-amf/model"
            #model_path = "../models/mx_models/v1_bn/model" #9
            #model_path = "../models/mx_models/model-100/model-v3/modelresave"
            epoch_num = 0 #9
            img_size = [112,112]
            FaceModel = mx_Face(model_path,epoch_num,img_size)
        elif faceconfig.caffe_use:
            face_p = "../models/sphere/sph_2.prototxt"
            face_m = "../models/sphere/sph20_ms_4v5.caffemodel"
            #face_p2 = "../models/mx_models/mobile_model/face-mobile.prototxt"
            #face_m2 = "../models/mx_models/mobile_model/face-mobile.caffemodel"
            if config.feature_1024:
                out_layer = 'fc5_n'
            elif config.insight:
                out_layer ='fc1'
            else:
                out_layer ='fc5'
            FaceModel = FaceReg(face_p,face_m,out_layer)
        self.FaceModel = FaceModel
        self.label_file = label_file
        self.id_dir = id_dir
        self.DB_FT = Annoy_DB(faceconfig.feature_lenth)
        self.db_file = db_file
        self.save_dir = save_dir
        self.mkdb()
        self.db_cnt_dict = dict()
        self.dist_path = None
        self.img_name = None
        self.img_cnt = 0

    def get_label(self):
        f_r = open(self.label_file,'rb')
        self.label_dict = pickle.load(f_r)
        #print(l_dict.values())
        f_r.close()
    def extract_f1(self,path_,id_dir):
        img_path = os.path.join(id_dir,path_)
        img = cv2.imread(img_path)
        img_org = img
        if faceconfig.db_enhance:
            img = self.img_enhance(img)
        if img is None:
            feat = None
            print("read img faild, " + img_path)
        else:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            if faceconfig.torch_:
                img = img*1.0
                img = img/255
            elif faceconfig.caffe_use:
                img = (img-127.5)*0.0078125
            feat = self.FaceModel.extractfeature(img)
        return feat,img_org

    def extract_f2(self,img):
        if img is None:
            feat = None
            print("read img faild, ")
        else:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            if faceconfig.torch_:
                img = img*1.0
                img = img/255
            elif faceconfig.caffe_use:
                img = (img-127.5)*0.0078125
            feat = self.FaceModel.extractfeature(img)
        return feat
    
    def mkdb(self):
        self.db_names = []
        self.db_paths = []
        self.db_imgs = []
        db_in = open(self.db_file,'r')
        db_lines = db_in.readlines()
        item_ = 0
        for line_one in db_lines:
            line_one = line_one.strip()
            base_feat,db_img = self.extract_f1(line_one,self.id_dir)
            if base_feat is None:
                print("feature is None")
                continue
            one_name = line_one[:-4]
            self.db_names.append(one_name)
            face_db_dir = os.path.join(self.save_dir,one_name)
            self.db_paths.append(face_db_dir)
            '''
            if os.path.exists(face_db_dir):
                pass
            else:
                os.makedirs(face_db_dir)
            '''
            self.DB_FT.add_data(item_,base_feat)
            self.db_imgs.append(db_img)
            item_+=1
        print("begin do build db")
        #print(db_features.shape)
        #DB_FT = Face_DB(db_features)
        self.DB_FT.build_db()
        print("db over")
        #self.get_label(self.label_file)
        db_in.close()

    def get_datas(self,data_file,base_dir,base_id):
        data_in = open(data_file,'r')
        data_lines = data_in.readlines()
        idx_ = 0
        for querry_line in data_lines:
            querry_line = querry_line.strip()
            que_spl = querry_line.split("/")
            key_que = que_spl[0][2:]
            real_label = self.label_dict.setdefault(key_que,300)+base_id
            idx_+=1
            sys.stdout.write('\r>> deal with %d/%d' % (idx_,len(data_lines)))
            sys.stdout.flush()
            querry_feat, _ = self.extract_f1(querry_line,base_dir)
            if querry_feat is None:
                print("feature is None")
                continue
        data_in.close()

    def mk_dirs(self,dir_path):
        if os.path.exists(dir_path):
            pass
        else:
            os.makedirs(dir_path)

    def get_namebyindx(self,idx):
        img_name = self.db_names[idx] 
        img_path = self.db_paths[idx]
        db_img = self.db_imgs[idx]
        #pred_label = label_dict.setdefault(img_name,300)+base_id
        img_cnt = self.db_cnt_dict.setdefault(img_name,0)
        if int(img_cnt) == 0:
            self.mk_dirs(img_path)
        dist_path = os.path.join(img_path,img_name+"_"+str(img_cnt)+".jpg")
        id_path = os.path.join(img_path,img_name+".jpg")
        return img_name,db_img,dist_path,img_cnt,id_path

    def img_enhance(self,image):
        fgamma = faceconfig.gamma
        image_gamma = np.uint8(np.power(image/255.0,fgamma)*255.0)
        cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
        cv2.convertScaleAbs(image_gamma, image_gamma)
        return image_gamma

    def findNearPerson(self,image):
        if faceconfig.face_enhance:
            img = self.img_enhance(image)
        else:
            img = image
        img_org = img
        querry_feat = self.extract_f2(img)
        if querry_feat is None:
            return None,None,None,
        else:
            idx_list,distance_list = self.DB_FT.findNearest(querry_feat,3) 
            if faceconfig.debug:
                print("distance list ",distance_list)
            #print("distance 1: ",distance_list[0],distance_list[1])
            if distance_list[0] <= faceconfig.top1_distance and (distance_list[1] - distance_list[0]) >= faceconfig.confidence:
                if faceconfig.debug:
                    print("distance 1: ",distance_list[0],distance_list[1])
                idx = idx_list[0]
                self.img_name,db_img,self.dist_path,self.img_cnt,self.id_path = self.get_namebyindx(idx)
                #cv2.imwrite(dist_path,img) 
                #self.db_cnt_dict[img_name] = img_cnt+1 
                re_person = self.img_name
                #print("pred 1",re_person)
            elif distance_list[0] <= 1.5 and (distance_list[2] - distance_list[0]) >= 100:
                #print("distance 2: ",distance_list[0],distance_list[2])
                idx = idx_list[0]
                self.img_name,db_img,self.dist_path,self.img_cnt,self.id_path = self.get_namebyindx(idx)
                re_person = self.img_name
                print("pred 2",re_person)
            else:
                re_person = None
                db_img = None
            self.db_img = db_img
            return re_person,db_img,img_org
    def saveperson(self,img):
        cv2.imwrite(self.dist_path,img)
        if self.db_cnt_dict[self.img_name] == 0:
            cv2.imwrite(self.id_path,self.db_img)
        self.db_cnt_dict[self.img_name] = self.img_cnt+1 

def img_crop(img,bbox,w,h):
    x1 = int(max(bbox[0],0))
    y1 = int(max(bbox[1],0))
    x2 = int(min(bbox[2],w))
    y2 = int(min(bbox[3],h))
    cropimg = img[y1:y2,x1:x2,:]
    return cropimg

def img_crop2(img,bboxes,crop_size,imgw,imgh):
    '''
    img: frame
    bboxes: [x1,y1,x2,y2,score,p1,...,p10],[],...
    '''
    crop_imgs = []
    for bbox in bboxes:
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        if videoconfig.det_box_widen:
            boxw = x2-x1
            boxh = y2-y1
            x1 = int(max(0,int(x1-0.2*boxw)))
            y1 = int(max(0,int(y1-0.2*boxh)))
            x2 = int(min(imgw,int(x2+0.2*boxw)))
            y2 = int(min(imgh,int(y2+0.2*boxh)))
        cropimg = img[y1:y2,x1:x2,:]
        cropimg = cv2.resize(cropimg,(crop_size[1],crop_size[0]))
        crop_imgs.append(cropimg)
    return crop_imgs

def img_crop3(img,bboxes,imgw,imgh):
    '''
    img: frame
    bboxes: [x1,y1,x2,y2,score,p1,...,p10],[],...
    '''
    crop_imgs = []
    for bbox in bboxes:
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        boxw = x2-x1
        boxh = y2-y1
        x1 = int(max(0,int(x1-0.2*boxw)))
        y1 = int(max(0,int(y1-0.2*boxh)))
        x2 = int(min(imgw,int(x2+0.2*boxw)))
        y2 = int(min(imgh,int(y2+0.2*boxh)))
        cropimg = img[y1:y2,x1:x2,:]
        crop_imgs.append(cropimg)
    return crop_imgs
     
def label_show(img,rectangles):
    for rectangle in rectangles:
        #print(map(int,rectangle[5:]))
        score_label = str("{:.2f}".format(rectangle[4]))
        #score_label = str(1.0)
        cv2.putText(img,score_label,(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        cv2.rectangle(img,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
        if len(rectangle) > 5:
            if detconfig.x_y:
                for i in range(5,15,2):
                    cv2.circle(img,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
            else:
                rectangle = rectangle[5:]
                for i in range(5):
                    cv2.circle(img,(int(rectangle[i]),int(rectangle[i+5])),2,(0,255,0))
    
def sort_box(boxes_or):
    boxes = np.array(boxes_or)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = area.argsort()[::-1]
    #print(I)
    #print(boxes_or[0])
    idx = map(int,I)
    return boxes[idx[:]].tolist()

def main(file_in,db_file,id_dir,save_dir,frame_num,out_file):
    if file_in is None:
        v_cap = cv2.VideoCapture(0)
    else:
        v_cap = cv2.VideoCapture(file_in)
    min_size = 24
    base_name = "test"
    threshold = np.array([0.8,0.9,0.95])
    if os.path.exists(save_dir):
        pass
    else:
        os.makedirs(save_dir)
    if os.path.isfile(out_file):
        record_w = open(out_file,'w')
    else:
        print("the output file is not exist: ",out_file)
    detect_model = MTCNNDet(min_size,threshold) 
    facereg_model = DB_Reg(db_file,id_dir,save_dir)
    #model_path = "../models/haarcascade_frontalface_default.xml"
    #detect_model = FaceDetector_Opencv(model_path)
    if faceconfig.mx_:
        crop_size = [112,112]
    else:
        crop_size = [112,96]
    idx_cnt = 0
    record_w.write("crop size is: %s \n" % ("\t".join([str(x) for x in crop_size])))
    person_name_dict = dict()
    cv2.namedWindow("gallery")
    cv2.namedWindow("querry")
    cv2.namedWindow("video")
    cv2.namedWindow("src")
    cv2.moveWindow("gallery",100,10)
    cv2.moveWindow("querry",500,10)
    cv2.moveWindow("video",1400,10)
    cv2.moveWindow("src",650,10)
    show_h = crop_size[0]
    show_w = crop_size[1]
    id_prob_show = np.zeros([6*show_h,3*show_w,3],dtype=np.uint8)
    if not v_cap.isOpened():
        print("field to open video")
    else:
        if file_in is not None:
            print("video frame num: ",v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_num = v_cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if total_num > 100000:
                total_num = 30000
        else:
            total_num = 100000
        record_w.write("the video has total num: %d \n" % total_num)
        if frame_num is not None:
            v_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        fram_w = v_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        fram_h = v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fram_cnt = 0
        while v_cap.isOpened():
            ret,frame = v_cap.read()
            fram_cnt+=1
            sys.stdout.write("\r>> deal with %d / %d" % (fram_cnt,total_num))
            sys.stdout.flush()
            if ret: 
                t = time.time()
                rectangles = detect_model.detectFace(frame)
                t_det = time.time() - t
            else:
                continue
            if len(rectangles)> 0:
                #rectangles = sort_box(rectangles)
                if faceconfig.time:
                    print("one frame detect time cost {:.3f} ".format(t_det))
                if detconfig.crop_org:
                    for bbox_one in rectangles:
                        idx_cnt+=1
                        img_out = img_crop(frame,bbox_one,fram_w,fram_h)
                        savepath = os.path.join(save_dir,base_name+'_'+str(idx_cnt)+".jpg")
                        #savepath = os.path.join(save_dir,line_1)
                        img_out = cv2.resize(img_out,(96,112))
                        #cv2.imwrite(savepath,img_out)
                else:
                    points = np.array(rectangles)
                    points = points[:,5:]
                    points_list = points.tolist()
                    #crop_imgs = Align_Image.extract_image_chips(frame,points_list)
                    if videoconfig.det_box_widen:
                        crop_imgs = img_crop2(frame,rectangles,crop_size,fram_w,fram_h)
                    else:
                        crop_imgs = alignImg(frame,crop_size,points_list)
                    if videoconfig.box_widen:
                        crop_widen_imgs = img_crop3(frame,rectangles,fram_w,fram_h)
                        assert len(crop_imgs) == len(crop_widen_imgs), "boxes widen is not equal the align"
                    if len(crop_imgs) == 0:
                        continue
                    for crop_id,img_out in enumerate(crop_imgs):
                        idx_cnt+=1
                        #savepath = os.path.join(save_dir,base_name+'_'+str(idx_cnt)+".jpg")
                        #img_out = cv2.resize(img_out,(96,112))
                        #cv2.imshow("test",img_out)
                        #cv2.imwrite(savepath,img_out)
                        t = time.time()
                        person_name, db_img,img_en = facereg_model.findNearPerson(img_out)
                        t_reg = time.time() - t
                        if person_name is not None:
                            if faceconfig.time:
                                print("a face recognize time cost {:.3f} ".format(t_reg))
                            person_cnt = person_name_dict.setdefault(person_name,fram_cnt)
                            if fram_cnt - person_cnt <= faceconfig.frame_interval:
                                if videoconfig.box_widen:
                                    facereg_model.saveperson(crop_widen_imgs[crop_id])
                                else:
                                    facereg_model.saveperson(img_out)
                                #person_name_dict[person_name] = person_cnt+1
                            person_name_dict[person_name] = person_cnt+1
                            #db_img = cv2.cvtColor(db_img,cv2.COLOR_RGB2BGR)
                            if faceconfig.debug:
                                print("crop img",img_out.shape,img_out[0,0:10,0])
                            id_prob_show[:show_h,:show_w,:] = db_img
                            #id_prob_show[:112,192:,:] = img_out
                            id_prob_show[:show_h,2*show_w:,:] = img_en
                            #cv2.imshow("querry",id_prob_show[:112,:96,:])
                            if faceconfig.debug:
                                print("show img id and crop ",id_prob_show[10,5:10,0],id_prob_show[10,200:210,0])
                            cv2.imshow("gallery",id_prob_show)
                            cv2.waitKey(50)
                            id_prob_show[show_h:,:,:] = id_prob_show[:-show_h,:,:]
                        cv2.imshow("querry",img_en)
                        #cv2.imshow("src",img_out)
                        if videoconfig.box_widen:
                            cv2.imshow("src",crop_widen_imgs[crop_id])
                        else:
                            cv2.imshow("src",img_out)
                label_show(frame,rectangles)
            else:
                #print("current frame no face ")
                pass
            cv2.imshow("video",frame)
            key_ = cv2.waitKey(10) & 0xFF
            if key_ == 27 or key_ == ord('q'):
                break
            if fram_cnt == total_num:
                break
    person_real = 0
    not_reg_name = []
    reg_img_cnt = 0
    for name in facereg_model.db_names:
        if name in facereg_model.db_cnt_dict.keys():
            if facereg_model.db_cnt_dict[name]:
                person_real+=1
                reg_img_cnt += facereg_model.db_cnt_dict[name]
            else:
                not_reg_name.append(name)
        else:
            not_reg_name.append(name)
    print("person ",person_real)
    print("total faces ",idx_cnt)
    print("reg faces ",reg_img_cnt)
    print("not reg person: ",not_reg_name)
    record_w.write("person recognized num: %d \n" % person_real)
    record_w.write("detected total faces: %d \n" % idx_cnt)
    record_w.write("recognize      faces: %d \n" % reg_img_cnt)
    record_w.write("in db not reg person: %s \n" % (" ".join([str(y) for y in not_reg_name])))
    record_w.close()
    v_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parm = args()
    videofile = parm.file_in
    db_file = parm.id_file
    id_dir = parm.base_dir
    save_dir = parm.save_dir
    frame_num = parm.frame_num
    output_file = parm.out_file
    main(videofile,db_file,id_dir,save_dir,frame_num,output_file)