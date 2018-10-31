# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/08/29 18:09
#project: Face recognize
#company: Senscape
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  detect face and extract features
####################################################
import sys
sys.path.append('../')
import cv2
import os
import numpy as np
from face_config import config
from demo.Detector import MTCNNDet

class ExtractFeatures(object):
    def __init__(self,model_):
        min_size = 24
        threshold = np.array([0.5,0.8,0.9])
        self.detect_model = MTCNNDet(min_size,threshold)
        self.face_model = model_

    def Img_enhance(self,image,fgamma=1.5):
        image_gamma = np.uint8(np.power((np.array(image)/255.0),fgamma)*255.0)
        cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
        cv2.convertScaleAbs(image_gamma, image_gamma)
        return image_gamma

    def detect_face2img(self,img,detect_model):
        '''
        img: image data input
        detect_model: used to detect faces model
        '''
        #img = img_ratio(img,320)
        rectangles = self.detect_model.detectFace(img)
        if len(rectangles)> 0:
            if len(rectangles)>1 :
                rectangles = self.sort_box(rectangles)
            img_out = self.img_crop(img,rectangles[0])
            #img_out = cv2.resize(img_out,(112,112))
            if config.show_img:
                img_show = img.copy()
                self.label_show(img_show,rectangles)
                cv2.imshow("test",img_show)
                cv2.imshow("crop",img_out)
                cv2.waitKey(1000)
        else:
            img_out = None
        return img_out

    def img_crop(self,img,bbox):
        imgh,imgw,imgc = img.shape
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        if config.box_widen:
            boxw = x2-x1
            boxh = y2-y1
            x1 = max(0,int(x1-0.2*boxw))
            y1 = max(0,int(y1-0.1*boxh))
            x2 = min(imgw,int(x2+0.2*boxw))
            #y2 = min(imgh,int(y2+0.1*boxh))
        cropimg = img[y1:y2,x1:x2,:]
        return cropimg

    def label_show(self,img,rectangles):
        for rectangle in rectangles:
            score_label = str("{:.2f}".format(rectangle[4]))
            #score_label = str(1.0)
            cv2.putText(img,score_label,(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
            cv2.rectangle(img,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
            if len(rectangles) > 5:
                for i in range(5,15,2):
                    cv2.circle(img,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))

    def sort_box(self,boxes_or):
        boxes = np.array(boxes_or)
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        area = np.multiply(x2-x1+1, y2-y1+1)
        I = area.argsort()[::-1]
        idx = map(int,I)
        return boxes[idx[:]].tolist()

    def img_ratio(self,img,img_h):
        h,w,c = img.shape
        ratio_ = float(h) / float(w)
        img_w = img_h / ratio_
        img_out = cv2.resize(img,(int(img_w),int(img_h)))
        return img_out 

    def extract_f(self,path_,base_dir):
        img_path = os.path.join(base_dir,path_)
        img = cv2.imread(img_path)
        if img is None:
            feat = None
            print("read img faild, " + img_path)
        else:
            if config.db_enhance:
                img = self.Img_enhance(img,config.gamma)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            if config.torch_:
                img = img*1.0
                img = img/255
            elif config.caffe_use:
                img = (img-127.5)*0.0078125
            feat = self.face_model.extractfeature(img)
            '''
            if config.norm:
                fea_std = np.std(feat)
                fea_mean = np.mean(feat)
                feat = (feat-fea_mean)/fea_std
            '''
        #if config.debug:
            #print("feature ",feat[:5])
        return feat

    def extract_f2(self,path_,base_dir):
        img_path = os.path.join(base_dir,path_)
        img = cv2.imread(img_path)
        if img is None:
            feat = None
            print("read img faild, " + img_path)
        else:
            if config.face_enhance:
                img = self.Img_enhance(img,config.gamma)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            if config.torch_:
                img = img*1.0
                img = img/255
            elif config.caffe_use:
                img = (img-127.5)*0.0078125
            feat = self.face_model.extractfeature(img)
            '''
            if config.norm:
                fea_std = np.std(feat)
                fea_mean = np.mean(feat)
                feat = (feat-fea_mean)/fea_std
            '''
        #if config.debug:
         #   print("feature ",feat[:5])
        return feat

    def extract_f3(self,path_,base_dir):
        img_path = os.path.join(base_dir,path_)
        img = cv2.imread(img_path)
        if img is None:
            feat = None
            print("read img faild, " + img_path)
        else:
            if config.face_enhance:
                img = self.Img_enhance(img,config.gamma)
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img_crop = self.detect_face2img(img,self.detect_model)
            if img_crop is None:
                img_crop = img
            if config.torch_:
                img_crop = img_crop*1.0
                img_crop = img_crop/255
            elif config.caffe_use:
                img_crop = (img_crop-127.5)*0.0078125
            feat = self.face_model.extractfeature(img_crop)
            '''
            if config.norm:
                fea_std = np.std(feat)
                fea_mean = np.mean(feat)
                feat = (feat-fea_mean)/fea_std
            '''
        #if config.debug:
         #   print("feature ",feat[:5])
        return feat