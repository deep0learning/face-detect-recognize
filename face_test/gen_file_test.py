# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2017/08/04 14:09
#project: Face recognize
#company: Senscape
#rversion: 0.1
#tool:   python 2.7
#modified:2018/06/14 09:24
#description  generate txt file,for example: img1.png  img1.png img2.png  ... img_n.png
####################################################
import numpy as np 
import os
import sys
import argparse
import shutil
import cv2
import pickle
import pylab as pl
import matplotlib.pyplot as plt
import string
from face_config import config

def parms():
    parser = argparse.ArgumentParser(description='gen img lists for confusion matrix test')
    parser.add_argument('--img-dir',type=str,dest="img_dir",default='./',\
                        help='the directory should include 2 more Picturess')
    parser.add_argument('--dir2-in',type=str,dest="dir2_in",default='./',\
                        help='the directory should include 2 more Picturess')
    parser.add_argument('--file-in',type=str,dest="file_in",default="train.txt",\
                        help='img paths saved file')
    parser.add_argument('--save-dir',type=str,dest="save_dir",default='./',\
                        help='img saved dir')
    parser.add_argument('--base-id',type=int,dest="base_id",default=0,\
                        help='img id')
    parser.add_argument('--out-file',type=str,dest="out_file",default="train.txt",\
                        help='out img paths saved file')
    parser.add_argument('--cmd-type',type=str,dest="cmd_type",default="None",\
                        help='which code to run: gen_label_pkl, gen_label,merge,gen_filepath_1dir,save_idimgfrom_txt,\
                        hist, imgenhance,gen_idimgfrom_dir,gen_filepath_2dir')
    parser.add_argument('--file2-in',type=str,dest="file2_in",default="train2.txt",\
                        help='label files')
    parser.add_argument('--hist-max',type=float,dest="hist_max",default=None,\
                        help='the max number in hist bin')
    parser.add_argument('--hist-bin',type=int,dest="hist_bin",default=20,\
                        help='the bin number')
    return parser.parse_args()

def get_from_dir(img_dir):
    '''
    img_dir: saved images path
            "img_dir/image1.jpg"
    return: confumax parten image path txtfile
            " img1.jpg  img2.jpg ... imgn.jpg"
    '''
    out_file = open('confuMax_test.txt','w')
    directory = img_dir
    filename = []
    for root,dirs,files in os.walk(directory):
        for file_1 in files:
            print("file:", file_1)
            filename.append(file_1)
    filename = np.sort(filename)
    print(len(filename))
    print(filename)
    for i in range(0,len(filename)):
        #print(len(filename))
        #out_file.write("{},".format(filename[i]))
        if i < len(filename):
            out_file.write("{},".format(filename[i]))
            #print(filename)
            for j in range(0,len(filename)):
                if j < len(filename)-1:
                    out_file.write("{},".format(filename[j]))
                else:
                    out_file.write("{}".format(filename[j]))
            out_file.write("\n")
    out_file.close()


def get_from_txt(file_in):
    '''
    file_in: saved images path
            "image1.jpg"
    return: confumax parten image path txtfile
            " img1.jpg  img2.jpg ... imgn.jpg"
    '''
    f_in = open(file_in,'r')
    f_out = open('confuMax_test.txt','w')
    filenames = f_in.readlines()
    filenames = np.sort(filenames)
    print("total ",len(filenames))
    #print(filenames[0])
    for i in range(len(filenames)):
        if i < len(filenames):
            f_out.write("{},".format(filenames[i].strip()))
            #print(filename)
            for j in range(0,len(filenames)):
                if j < len(filenames)-1:
                    f_out.write("{},".format(filenames[j].strip()))
                else:
                    f_out.write("{}".format(filenames[j].strip()))
            f_out.write("\n")
    f_in.close()
    f_out.close()

def generate_list_from_dir(dirpath,out_file):
    '''
    dirpath: saved images path
            "dirpath/id_num/image1.jpg"
    return: images paths txtfile
            "id_num/img1.jpg"
    '''
    f_w = open(out_file,'w')
    files = os.listdir(dirpath)
    total_ = len(files)
    print("total id ",len(files))
    idx =0
    file_name = []
    total_cnt = 0
    for file_cnt in files:
        img_dir = os.path.join(dirpath,file_cnt)
        imgs = os.listdir(img_dir)
        idx+=1
        sys.stdout.write("\r>>convert  %d/%d" %(idx,total_))
        sys.stdout.flush()
        for img_one in imgs:
            if config.img_lenth8 and len(img_one)<=8:
                continue
            img_path = os.path.join(file_cnt,img_one)
            total_cnt+=1
            f_w.write("{}\n".format(img_path))
    print("total img ",total_cnt)
    cnt = 0
    f_w.close()

def read_1(file_in):
    f_= open(file_in,'r')
    lines = f_.readlines()
    print(lines[-1])
    f_.close()

def gen_filefromdir(base_dir,txt_file):
    '''
    base_dir: saved images path
        "base_dir/image.jpg"
    return: "image1.jpg"
    '''
    f_w = open(txt_file,'w')
    files = os.listdir(base_dir)
    total_ = len(files)
    print("total id ",len(files))
    for file_cnt in files:
        f_w.write("{}\n".format(file_cnt))
    f_w.close()

def merge2txt(file1,file2):
    '''
    #parm:
           file1: id_file saved id images path
           file2: image_file saved train images path
        return: " id_num1 image1.jpg  ... imagen.jpg"
    '''
    f1 = open(file1,'r')
    f2 = open(file2,'r')
    f_out = open("db_test.txt",'w')
    id_files = f1.readlines()
    imgs = f2.readlines()
    for img_gal in imgs:
        f_out.write("{},".format(img_gal.strip()))
        for j in range(len(id_files)):
            if j < len(id_files)-1:
                f_out.write("{},".format(id_files[j].strip()))
            else:
                f_out.write("{}".format(id_files[j].strip()))
        f_out.write("\n")
    f1.close()
    f2.close()
    f_out.close()
    print("over")

def gen_dirfromtxt(file_p,base_dir,save_dir):
    '''
    #parms:
            file_p: saved images path;
                "id_num/image.jpg"
            base_dir: saved images basedir;
            saved_dir: images copied to dir
        return: images will copy to several documents;
            " 0/ 1/ 2/ "
    '''
    f_in = open(file_p,'r')
    file_lines = f_in.readlines()
    file_cnt = 0
    total_cnt = 0
    print("total id ",len(file_lines))
    for line_one in file_lines: 
        if total_cnt % 10000==0:
            dist_dir = os.path.join(save_dir,str(file_cnt))
            if not os.path.exists(dist_dir):
                os.makedirs(dist_dir)
            file_cnt+=1
            print(file_cnt)
        total_cnt+=1
        line_one = line_one.strip()
        org_path = os.path.join(base_dir,line_one)
        dist_path = os.path.join(dist_dir,line_one)
        shutil.copyfile(org_path,dist_path)
    f_in.close()


def generate_label_from_dir(dirpath):
    '''
    #param: 
        dirpath: saved images classified by server;
            " 11+id_num/img.jpg"
        return: "img_path label"
    '''
    f_w = open("prison_train.txt",'w')
    #f2_w = open("prison_val.txt",'w')
    files = os.listdir(dirpath)
    files = np.sort(files)
    total_ = len(files)
    print("total id ",len(files))
    idx =0
    cnt = 0
    for file_cnt,line_one in enumerate(files):
        img_dir = os.path.join(dirpath,line_one)
        imgs = os.listdir(img_dir)
        #cnt = 0
        idx+=1
        sys.stdout.write("\r>>convert  %d/%d" %(idx,total_))
        sys.stdout.flush()
        label_ = int(file_cnt)
        for img_one in imgs:
            if len(img_one) <=8:
                continue
            img_path = os.path.join(img_dir,img_one)
            f_w.write("{} {}\n".format(img_path,label_))
            '''
            if cnt ==100000:
                #f2_w.write("{} {}\n".format(img_path,label_))
                f2 = 0
            else:
                f_w.write("{} {}\n".format(img_path,label_))
            '''
            cnt+=1
    f_w.close()
    print("total ",cnt)
    #f2_w.close()

def get_labelfromtxt(file_in):
    '''
    parm: file_in: saved image path and label for train
         "base_dir/11+id_num/image.jpg  label"
         return: "id_num: label"
    '''
    f_r = open(file_in,'r')
    files = f_r.readlines()
    label_dict = dict()
    for line_in in files:
        line_in = line_in.strip()
        line_s = line_in.split(' ')
        label = int(line_s[1])
        id_num = line_s[0].split('/')[-2]
        id_num = id_num[2:]
        label_dict[id_num] = label
    f_w = open("prison_label_10778.pkl",'wb')
    #np.save("prison_label_218.npy",label_dict)
    pickle.dump(label_dict,f_w)
    print("keys ",label_dict.keys())
    print("values ",label_dict.values())
    f_r.close()
    f_w.close()

def get_labelfromdir(id_dir,out_file):
    '''
    parm: id_dir: saved id images; 
            "id_num.jpg"
        return: "id_num: label"
    '''
    id_files = os.listdir(id_dir)
    id_files = np.sort(id_files)
    label_dict = dict()
    for label, line_one in enumerate(id_files):
        id_num = line_one[:-4]
        #id_num = line_one[2:]
        label_dict[id_num] = label
    f_w = open(out_file,'wb')
    pickle.dump(label_dict,f_w)
    print("keys ",label_dict.keys())
    print("values ",label_dict.values())
    f_w.close()

def load_npy(file_in):
    #np.save(f,a,b,sinc=c)
    #label_k = np.load(file_in)
    #label_k['arr_0']
    #print("label_k ",np.shape(label_k))
    f_r = open(file_in,'rb')
    l_dict = pickle.load(f_r)
    print(l_dict.values())
    f_r.close()

def remove_f(path):
    cnt_img=0
    if not os.path.exists(path):
        print("path is not exist")
    else:
        for one_file in os.listdir(path):
            img_path = os.path.join(path,one_file)
            if not os.path.isfile(img_path):
                print("img is not exist")
            else:
                img = cv2.imread(img_path)
                if img is None:
                    print("img none ",one_file)
                    os.remove(img_path)
                else:
                    cnt_img +=1
                    continue
    print("img: ",cnt_img)

def generate_global_label_from_dir(label_file,dirpath,base_id,out_file):
    '''
    param: label_file- saved *.pkl, including a dict; 
            "id_num: label"
        dirpath: saved images classified by server;
            " 11+id_num/img.jpg"
        return: "img_path label"
    '''
    #f_w = open("prison_train_global3.txt",'w')
    f_w = open(out_file,'w')
    label_f = open(label_file,'rb')
    label_dict = pickle.load(label_f)
    files = os.listdir(dirpath)
    files = np.sort(files)
    total_ = len(files)
    print("total id ",len(files))
    idx =0
    cnt = 0
    for file_cnt,line_one in enumerate(files):
        img_dir = os.path.join(dirpath,line_one)
        imgs = os.listdir(img_dir)
        #cnt = 0
        idx+=1
        sys.stdout.write("\r>>convert  %d/%d" %(idx,total_))
        sys.stdout.flush()
        #label_ = int(file_cnt)
        label_ = int(label_dict[line_one[2:]])+base_id
        for img_one in imgs:
            if len(img_one) <=8:
                continue
            img_path = os.path.join(img_dir,img_one)
            f_w.write("{} {}\n".format(img_path,label_))
            '''
            if cnt ==100000:
                #f2_w.write("{} {}\n".format(img_path,label_))
                f2 = 0
            else:
                f_w.write("{} {}\n".format(img_path,label_))
            '''
            cnt+=1
    f_w.close()
    label_f.close()
    print("total ",cnt)

def merge2trainfile(file1,file2,file_out):
    '''
    file1: saved image  paths
    file2: saved image paths
    return: "file1 file2" 
    '''
    f1 = open(file1,'r')
    f2 = open(file2,'r')
    f_out = open(file_out,'w')
    id_files = f1.readlines()
    imgs = f2.readlines()
    for line_one in id_files:
        f_out.write(line_one.strip())
        f_out.write("\n")
    for idx,line_one in enumerate(imgs):
        if idx == 10000:
            break
        f_out.write(line_one.strip())
        f_out.write("\n")
    f1.close()
    f2.close()
    f_out.close()
    print("over")

def rm_imgfromtxt(base_dir,file_in):
    '''
    basedir: images saved dir, eg: root/
    file_in: saved images path, eg: 100/1.jpg
    fun: delete the spacial parten image, eg: 11706.jpg
    '''
    f_r = open(file_in,'r')
    file_lines = f_r.readlines()
    for line_one in file_lines:
        #print(line_one)
        line_one = line_one.strip()
        line_s = line_one.split('/')
        if len(line_s[1]) <=8:
            img_path  = os.path.join(base_dir,line_one)
            if os.path.exists(img_path):
                print(img_path)
                os.remove(img_path)
                #shutil.move()
            else:
                print("not exist ",img_path)
        else:
            continue
    print("over") 

def save_cropface(file_in,base_dir,save_dir):
    '''
    file_in: saved img paths in txt file
    base_dir: images input dir
    saved_dir: images output dir
    fun: save the input images to 112x96 size
    '''
    f_in = open(file_in,'r')
    files_in = f_in.readlines()
    img_size = (96,112)
    for line_one in files_in:
        line_one = line_one.strip()
        in_path = os.path.join(base_dir,line_one)
        img = cv2.imread(in_path)
        if img is not None:
            img = cv2.resize(img,img_size)
            out_path = os.path.join(save_dir,line_one)
            cv2.imwrite(out_path,img)
    print("over")
    f_in.close()

def plt_tpr2fpr(tpr_file,fpr_file):
    '''
    tpr_file: saved datas TPR
    fpr_file: saved datas FPR
    the 2 input file data is corresponding,eg: tpr-fpr: 0.8-0.1
    '''
    data1 = np.loadtxt(tpr_file)  
    fdata1 = np.loadtxt(fpr_file)
    print(fdata1[:,0])
    print(data1[:,0])
    plot1 = pl.plot(fdata1[:,0], data1[:,0],label='ful_dataset',pickradius=0.01,markersize=12)# use pylab to plot x and y : Give your plots names
    pl.title('TPR vs FPR')# give plot a title
    pl.xlabel('FPR')# make axis labels
    pl.ylabel('TPR')
    pl.xlim(0.0, 0.01)# set axis limits
    pl.ylim(0.0, 1.0)
    pl.legend(loc='best')
    pl.show()# show the plot on the screens

def plt_histgram(file_in,file_out,distance,num_bins=20):
    '''
    file_in: saved distances txt file
    file_out: output bins and 
    '''
    out_name = file_out
    input_file=open(file_in,'r')
    out_file=open(file_out,'w')
    data_arr=[]
    print(out_name)
    out_list = out_name.strip()
    out_list = out_list.split('/')
    out_name = "./output/"+out_list[-1][:-4]+".png"
    print(out_name)
    for line in input_file.readlines():
        line = line.strip()
        temp=string.atof(line)
        data_arr.append(temp)
    data_in=np.asarray(data_arr)
    if distance is None:
        max_bin = np.max(data_in)
    else:
        max_bin = distance
    datas,bins,c=plt.hist(data_in,num_bins,range=(0.0,max_bin),normed=1,color='blue',cumulative=1)
    #a,b,c=plt.hist(data_in,num_bins,normed=1,color='blue',cumulative=1)
    plt.title('histogram')
    plt.savefig(out_name, format='png')
    plt.show()
    for i in range(num_bins):
        out_file.write(str(datas[i])+'\t'+str(bins[i])+'\n')
    input_file.close()
    out_file.close()

def preprocess(filename):
    '''
    enhancement the image using filters
    '''
    image = cv2.imread(filename)
    img_src = image
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("src")
    cv2.namedWindow("hist")
    cv2.namedWindow("laplace")
    cv2.namedWindow("log")
    cv2.namedWindow("gamma")
    cv2.moveWindow("src",100,10)
    cv2.moveWindow("hist",250,10)
    cv2.moveWindow("laplace",500,10)
    cv2.moveWindow("log",650,10)
    cv2.moveWindow("gamma",800,10)
    
#    hist enhance
    image_equal = cv2.equalizeHist(image_gray)
    r,g,b = cv2.split(image)
    r1 = cv2.equalizeHist(r)
    g1 = cv2.equalizeHist(g)
    b1 = cv2.equalizeHist(b)
    image_equal_clo = cv2.merge([r1, g1, b1])
#   laplace enhance
    kernel = np.array([ [-1, -1, -1],  
                    [-1,  9, -1],  
                    [-1, -1, -1] ]) 
    image_lap = cv2.filter2D(image,cv2.CV_8UC3 , kernel)
#    log enhance
    image_log = np.uint8(np.log(np.array(image) +1))    
    cv2.normalize(image_log, image_log,0,255,cv2.NORM_MINMAX)
#   convert unit8 datatype
    cv2.convertScaleAbs(image_log,image_log)
#   gamma transform
    #fgamma = 1.5
    fgamma = 0.5
    image_gamma = np.uint8(np.power((np.array(image)/255.0),fgamma)*255.0)
    cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
    cv2.convertScaleAbs(image_gamma, image_gamma)
    #show
    cv2.imshow("src",img_src)
    cv2.imshow("hist",image_equal_clo)
    cv2.imshow("laplace",image_lap)
    cv2.imshow("log",image_log)
    cv2.imshow("gamma",image_gamma)
    key_ = cv2.waitKey(0) & 0xFF
    if key_ == 27 or key_ == ord('q'):
        cv2.destroyAllWindows()

from PIL import Image
from PIL import ImageEnhance
 
def pil_process(filename):
    image = Image.open(filename)
    image.show()
    #亮度增强
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    #image_brightened.show()
    
    #色度增强
    enh_col = ImageEnhance.Color(image)
    color = 1.5
    image_colored = enh_col.enhance(color)
    #image_colored.show()
    
    #对比度增强
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    image_contrasted.show()
    
    #锐度增强
    enh_sha = ImageEnhance.Sharpness(image)
    sharpness = 3.0
    image_sharped = enh_sha.enhance(sharpness)
    #image_sharped.show()

def gen_idimg_from_dir(dirpath,saved_dir):
    '''
    dirpath: saved images path
            "dirpath/id_num/image1.jpg"
    return: images paths txtfile
            "saved_dir/id_num.jpg"
    '''
    files = os.listdir(dirpath)
    total_ = len(files)
    print("total id ",len(files))
    total_cnt = 0
    for file_cnt in files:
        img_dir = os.path.join(dirpath,file_cnt)
        dist_img = file_cnt[2:]+".jpg"
        dist_img_path = os.path.join(saved_dir,dist_img)
        imgs = os.listdir(img_dir)
        img_num = len(imgs)
        #print("len",img_num)
        if img_num >1:
            sel_num = np.random.randint(img_num-1)
        else:
            sel_num = img_num-1
            print("this is 1:",file_cnt)
        sel_img = imgs[sel_num]
        org_img_path = os.path.join(img_dir,sel_img)
        if os.path.exists(org_img_path):
            total_cnt+=1
            shutil.copyfile(org_img_path,dist_img_path)
        else:
            print("img path is not exist",org_img_path)
    print("total img ",total_cnt)

def compare_2dir(base_dir,dir2,saved_dir):
    '''
    dir1: saved imgs dir
            "dir1/id_num/image1.jpg"
    dir2: saved imgs dir
            "dir2/id_num/image1.jpg"
    saved_dir: resave images from dir2  not in dir1
    '''
    def make_dirs(dir_path):
        if os.path.exists(dir_path):
            pass 
        else:
            os.makedirs(dir_path)
    make_dirs(saved_dir)
    tpr = 0
    idx_ = 0
    dir1_files = os.listdir(base_dir)
    total_file1 = len(dir1_files)
    dir1_id_dict = dict()
    for id_num in dir1_files:
        dir1_id_dict[id_num] = dir1_id_dict.setdefault(id_num,[])
        img_dir = os.path.join(base_dir,id_num)
        img_files = os.listdir(img_dir)
        for item_img in img_files:
            item_img = item_img.strip()
            if len(item_img)<=8:
                continue
            dir1_id_dict[id_num].append(item_img)
    dir2_files = os.listdir(dir2)
    total_file2 = len(dir2_files)
    dir2_id_cnt_dict = dict()
    for id2_num in dir2_files:
        idx_+=1
        sys.stdout.write('\r>> deal with %d/%d' % (idx_,total_file2))
        sys.stdout.flush()
        id_cnt = dir2_id_cnt_dict.setdefault(id2_num,0)
        img2_dir = os.path.join(dir2,id2_num)
        img2_files = os.listdir(img2_dir)
        if id2_num in dir1_id_dict.keys():
            id_files = dir1_id_dict[id2_num]
        else:
            id_files = []
        for item_img2 in img2_files:
            item_img2 = item_img2.strip()
            if item_img2 in id_files:
                tpr+=1
                dir2_id_cnt_dict[id2_num] = id_cnt+1
            else:
                failed_img_dir = os.path.join(saved_dir,id2_num)
                make_dirs(failed_img_dir)
                failed_img_path = os.path.join(failed_img_dir,item_img2)
                org_img_path = os.path.join(img2_dir,item_img2)
                shutil.copyfile(org_img_path,failed_img_path)
    not_reg_names = []
    for item_name in dir1_id_dict.keys():
        if item_name in dir2_id_cnt_dict.keys():
            if dir2_id_cnt_dict[item_name] >0:
                pass
            else:
                not_reg_names.append(item_name)
        else:
            not_reg_names.append(item_name)
    print("total id in file1: ",total_file1)
    print("total id in file2: ",total_file2)
    print("dir2 match imgs: ",tpr)
    print("dir2 not in dir1 names: ",not_reg_names)


if __name__ == "__main__":
    args = parms()
    txt_file = args.file_in
    #get_from_txt(txt_file)
    img_dir = args.img_dir
    save_dir = args.save_dir
    base_id = args.base_id
    file2_in = args.file2_in
    out_file = args.out_file
    cmd_type = args.cmd_type
    hist_max = args.hist_max
    hist_bin = args.hist_bin
    dir2 = args.dir2_in
    #read_1(txt_file)
    #gen_dirfromtxt(txt_file,img_dir,save_dir)
    #generate_label_from_dir(img_dir)
    #remove_f(img_dir)
    #get_labelfromtxt(txt_file)
    #load_npy(txt_file)
    if cmd_type == 'gen_label':
        generate_global_label_from_dir(txt_file,img_dir,base_id,out_file)
    elif cmd_type == 'merge':
        merge2trainfile(txt_file,file2_in,out_file)
    elif cmd_type == 'gen_label_pkl':
        get_labelfromdir(img_dir,out_file)
    elif cmd_type == 'gen_filepath_1dir':
        gen_filefromdir(img_dir,out_file)
    elif cmd_type == 'save_idimgfrom_txt':
        save_cropface(txt_file,img_dir,save_dir)
    elif cmd_type == 'hist':
        plt_histgram(txt_file,out_file,hist_max,hist_bin)
    elif cmd_type == 'imgenhance':
        preprocess(img_dir)
        #pil_process(img_dir)
    elif cmd_type == 'gen_idimgfrom_dir':
        gen_idimg_from_dir(img_dir,save_dir)
    elif cmd_type == 'gen_filepath_2dir':
        generate_list_from_dir(img_dir,out_file)
    elif cmd_type == 'compare_dir':
        compare_2dir(img_dir,dir2,save_dir)
    #rm_imgfromtxt(img_dir,txt_file)
