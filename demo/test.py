# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/06/12 14:09
#project: Face detect
#company: Senscape
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face detect testing caffe model
####################################################
import sys
sys.path.append('.')
sys.path.append('/home/lxy/caffe/python')
import os
os.environ['GLOG_minloglevel'] = '2'
import tools_matrix as tools
import caffe
import cv2
import numpy as np
import argparse
import time
from scipy.spatial import distance
from mtcnn_config import config
from Detector import FaceDetector_Opencv,MTCNNDet
import shutil
from align import Align_img , alignImg

def args():
    parser = argparse.ArgumentParser(description="mtcnn caffe")
    parser.add_argument('--file-in',type=str,dest='file_in',default='None',\
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
    parser.add_argument('--save-dir',type=str,dest='save_dir',default="./",\
                        help="images saved dir")
    parser.add_argument('--base-name',type=str,dest='base_name',default="videox",\
                        help="images saved dir")
    parser.add_argument('--cmd-type',type=str,dest='cmd_type',default="video",\
                        help="detect face from : video or txtfile")
    parser.add_argument('--save-dir2',type=str,dest='save_dir2',default=None,\
                        help="images saved dir")
    return parser.parse_args()


def evalu_img(imgpath,min_size):
    cv2.namedWindow("test")
    cv2.moveWindow("test",1400,10)
    threshold = np.array([0.5,0.5,0.9])
    detect_model = MTCNNDet(min_size,threshold)  
    img = cv2.imread(imgpath)    
    rectangles = detect_model.detectFace(img)
    draw = img.copy()
    for rectangle in rectangles:
        score_label = str("{:.2f}".format(rectangle[4]))
        cv2.putText(draw,score_label,(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
        for i in range(5,15,2):
            cv2.circle(draw,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
    cv2.imshow("test",draw)
    cv2.waitKey(0)
    #cv2.imwrite('test.jpg',draw)

def main():
    cv2.namedWindow("test")
    cv2.moveWindow("test",1400,10)
    threshold = [0.5,0.9,0.9]
    imgpath = "test2.jpg"
    parm = args()
    min_size = parm.min_size
    file_in = parm.file_in
    detect_model = MTCNNDet(min_size,threshold)
    if file_in is 'None':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(file_in)
    if not cap.isOpened():
        print("failed open camera")
        return 0
    else: 
        while 1:
            _,frame = cap.read()
            rectangles = detect_model.detectFace(frame)
            draw = frame.copy()
            if len(rectangles) > 0:
                for rectangle in rectangles:
                    score_label = str("{:.2f}".format(rectangle[4]))
                    cv2.putText(draw,score_label,(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
                    cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
                    if len(rectangle) > 5:
                        for i in range(5,15,2):
                            cv2.circle(draw,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
            cv2.imshow("test",draw)
            q=cv2.waitKey(10) & 0xFF
            if q == 27 or q ==ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()   

def save_cropfromtxt(file_in,base_dir,save_dir):
    '''
    file_in: images path recorded
    base_dir: images locate in 
    save_dir: detect faces saved in
    fun: saved id images, save name is the same input image
    '''
    f_ = open(file_in,'r')
    lines_ = f_.readlines()
    min_size = 24
    threshold = np.array([0.5,0.8,0.9])
    detect_model = MTCNNDet(min_size,threshold) 
    #model_path = "../models/haarcascade_frontalface_default.xml"
    #detect_model = FaceDetector_Opencv(model_path)
    def img_crop(img,bbox):
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
    idx_cnt = 0 
    def label_show(img,rectangles):
        for rectangle in rectangles:
            score_label = str("{:.2f}".format(rectangle[4]))
            #score_label = str(1.0)
            cv2.putText(img,score_label,(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
            cv2.rectangle(img,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
            if len(rectangles) > 5:
                for i in range(5,15,2):
                    cv2.circle(img,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
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
    def img_ratio(img,img_h):
        h,w,c = img.shape
        ratio_ = float(h) / float(w)
        img_w = img_h / ratio_
        img_out = cv2.resize(img,(int(img_w),int(img_h)))
        return img_out
    for line_1 in lines_:
        line_1 = line_1.strip()
        img_path = os.path.join(base_dir,line_1)
        img = cv2.imread(img_path) 
        img = img_ratio(img,320)
        line_s = line_1.split("/")  
        img_name = line_s[-1]
        new_dir = '/'.join(line_s[:-1]) 
        rectangles = detect_model.detectFace(img)
        if len(rectangles)> 0:
            idx_cnt+=1
            rectangles = sort_box(rectangles)
            img_out = img_crop(img,rectangles[0])
            #savepath = os.path.join(save_dir,str(idx_cnt)+".jpg")
            img_out = cv2.resize(img_out,(96,112))
            savepath = os.path.join(save_dir,line_1)
            '''
            savedir = os.path.join(save_dir,new_dir)
            if os.path.exists(savedir):
                savepath = os.path.join(savedir,img_name)
                shutil.copyfile(img_path,savepath)
            else:
                os.makedirs(savedir)
                savepath = os.path.join(savedir,img_name)
                shutil.copyfile(img_path,savepath)
            '''
            cv2.imwrite(savepath,img_out)
            cv2.waitKey(1000)
            #cv2.imwrite(savepath,img)
            label_show(img,rectangles)
        else:
            print("failed ",img_path)
        cv2.imshow("test",img)
        cv2.waitKey(10)


def save_cropfromvideo(file_in,base_name,save_dir,save_dir2):
    '''
    file_in: input video file path
    base_name: saved images prefix name
    save_dir: saved images path
    fun: saved detect faces to dir
    '''
    if file_in is None:
        v_cap = cv2.VideoCapture(0)
    else:
        v_cap = cv2.VideoCapture(file_in)
    min_size = 24
    threshold = np.array([0.7,0.8,0.95])
    detect_model = MTCNNDet(min_size,threshold) 
    #model_path = "../models/haarcascade_frontalface_default.xml"
    #detect_model = FaceDetector_Opencv(model_path)
    crop_size = [112,96]
    Align_Image = Align_img(crop_size)
    def mk_dirs(path):
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)
    def img_crop(img,bbox,w,h):
        x1 = int(max(bbox[0],0))
        y1 = int(max(bbox[1],0))
        x2 = int(min(bbox[2],w))
        y2 = int(min(bbox[3],h))
        cropimg = img[y1:y2,x1:x2,:]
        return cropimg
    def img_crop2(img,bbox,imgw,imgh):
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        if config.box_widen:
            boxw = x2-x1
            boxh = y2-y1
            x1 = int(max(0,int(x1-0.2*boxw)))
            y1 = int(max(0,int(y1-0.1*boxh)))
            x2 = int(min(imgw,int(x2+0.2*boxw)))
            y2 = int(min(imgh,int(y2+0.1*boxh)))
        cropimg = img[y1:y2,x1:x2,:]
        return cropimg
    idx_cnt = 0 
    def label_show(img,rectangles):
        for rectangle in rectangles:
            #print(map(int,rectangle[5:]))
            score_label = str("{:.2f}".format(rectangle[4]))
            #score_label = str(1.0)
            cv2.putText(img,score_label,(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
            cv2.rectangle(img,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
            if len(rectangle) > 5:
                if config.x_y:
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
    mk_dirs(save_dir)
    mk_dirs(save_dir2)
    if not v_cap.isOpened():
        print("field to open video")
    else:
        print("video frame num: ",v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_num = v_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fram_w = v_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        fram_h = v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fram_cnt = 0
        while v_cap.isOpened():
            ret,frame = v_cap.read()
            fram_cnt+=1
            if ret: 
                rectangles = detect_model.detectFace(frame)
            else:
                continue
            if len(rectangles)> 0:
                #rectangles = sort_box(rectangles)
                if config.crop_org:
                    for bbox_one in rectangles:
                        idx_cnt+=1
                        img_out = img_crop(frame,bbox_one,fram_w,fram_h)
                        savepath = os.path.join(save_dir,base_name+'_'+str(idx_cnt)+".jpg")
                        #savepath = os.path.join(save_dir,line_1)
                        img_out = cv2.resize(img_out,(96,112))
                        cv2.imwrite(savepath,img_out)
                else:
                    points = np.array(rectangles)
                    points = points[:,5:]
                    points_list = points.tolist()
                    #crop_imgs = Align_Image.extract_image_chips(frame,points_list)
                    crop_imgs = alignImg(frame,crop_size,points_list)
                    for box_idx,img_out in enumerate(crop_imgs):
                        idx_cnt+=1
                        savepath = os.path.join(save_dir,base_name+'_'+str(idx_cnt)+".jpg")
                        #img_out = cv2.resize(img_out,(96,112))
                        #cv2.imshow("test",img_out)
                        cv2.imwrite(savepath,img_out)
                        cv2.waitKey(50)
                        if config.box_widen:
                            savepath2 = os.path.join(save_dir2,base_name+'_'+str(idx_cnt)+".jpg")
                            img_widen = img_crop2(frame,rectangles[box_idx],fram_w,fram_h)
                            cv2.imwrite(savepath2,img_widen)
                            cv2.waitKey(50)
                        print("crop num,",idx_cnt)
                '''
                savedir = os.path.join(save_dir,new_dir)
                if os.path.exists(savedir):
                    savepath = os.path.join(savedir,img_name)
                    shutil.copyfile(img_path,savepath)
                else:
                    os.makedirs(savedir)
                    savepath = os.path.join(savedir,img_name)
                    shutil.copyfile(img_path,savepath)
                '''
                #cv2.imwrite(savepath,img)
                #label_show(frame,rectangles)
            else:
                #print("failed ")
                pass
            #cv2.imshow("test",frame)
            key_ = cv2.waitKey(10) & 0xFF
            if key_ == 27 or key_ == ord('q'):
                break
            if fram_cnt == total_num:
                break
    print("total ",idx_cnt)
    v_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #main()
    parm = args()
    p1 = parm.img_path1
    p2 = parm.img_path2
    f_in = parm.file_in
    base_dir = parm.base_dir
    save_dir = parm.save_dir
    base_name = parm.base_name
    cmd_type = parm.cmd_type
    save_dir2 = parm.save_dir2
    #base_name = parm.img_path1
    if cmd_type == 'txtfile':
        save_cropfromtxt(f_in,base_dir,save_dir)
    elif cmd_type == 'video':
        save_cropfromvideo(f_in,base_name,save_dir,save_dir2)
