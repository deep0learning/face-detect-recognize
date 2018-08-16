###############################################
#created by :  lxy
#Time:  2018/07/11 19:09
#project: Face recognize
#company: Senscape
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  gen hdf5 files for caffe train
####################################################
#!/usr/bin/env python
# coding=utf-8

import os
import sys
import cv2
import h5py
import argparse
import numpy as np
import random

def write_hdf5(file, data, label_class):
    # transform to np array
    data_arr = np.array(data, dtype = np.float32)
    # print data_arr.shape
    # if no swapaxes, transpose to num * channel * width * height ???
    # data_arr = data_arr.transpose(0, 3, 2, 1)
    label_class_arr = np.array(label_class, dtype = np.float32)
    with h5py.File(file, 'w') as f:
        f['data'] = data_arr
        f['label_class'] = label_class_arr
        f.close()

# list_file format:
# image_path | label_class 
def convert_dataset_to_hdf5(list_file, path_data, path_save,size_hdf5, tag):
    with open(list_file, 'r') as f:
        annotations = f.readlines()
    num = len(annotations)
    print ("%d pics in total" % num)
    random.shuffle(annotations)
    data = []
    label_class = []
    count_data = 0
    count_hdf5 = 0
    for line in annotations:
        line_split = line.strip().split()
        assert len(line_split) == 2
        path_full = os.path.join(path_data, line_split[0])
        datum = cv2.imread(path_full)
        classes = float(line_split[1])
        if datum is None:
            continue
        # normalization# BGR to RGB
        tmp = datum[:, :, 2]
        datum[:, :, 2] = datum[:, :, 0]
        datum[:, :, 0] = tmp
        datum = (datum - 127.5) * 0.0078125 # [0,255] -> [-1,1]
        #datum = np.swapaxes(datum, 0, 2)
        datum = np.transpose(datum,(2,0,1))
        print("train data shape ",np.shape(datum))
        data.append(datum)
        label_class.append(classes)
        count_data = count_data + 1
        sys.stdout.write('\r>> Converting image %d/%d' % (count_data, num))
        sys.stdout.flush()
        if 0 == count_data % size_hdf5:
            path_hdf5 = os.path.join(path_save, tag + str(count_hdf5) + ".h5")
            write_hdf5(path_hdf5, data, label_class)
            count_hdf5 = count_hdf5 + 1
            data = []
            label_class = []
        #print(count_data)
    # handle the rest
    if data:
        path_hdf5 = os.path.join(path_save, tag + str(count_hdf5) + ".h5")
        write_hdf5(path_hdf5, data, label_class)
    print("count_data: %d" % count_data)
    f.close()

def main():
    parser = argparse.ArgumentParser(description = 'Convert dataset to hdf5')
    parser.add_argument('--list-file',type=str,dest='list_file',default='prison_train_global.txt',
                        help = 'Special format list file')
    parser.add_argument('--img-dir',dest='img_dir', type=str,default='/home/lxy/',
                        help = 'Path to original dataset')
    parser.add_argument('--path-save',dest='path_save',type=str,default='./hdf5_train', help = 'Path to save hdf5')
    parser.add_argument('--size_hdf5', type = int,default=40960,
                        help = 'Batch size of hdf5, Default: 4096')
    parser.add_argument('--tag', type = str,default='train_',
                        help = 'Specify train, test or validation, Default: train_')
    #parser.set_defaults(size_hdf5 = 4096, tag = 'train_')
    args = parser.parse_args()
    list_file = args.list_file
    path_data = args.img_dir
    path_save = args.path_save
    size_hdf5 = args.size_hdf5
    tag = args.tag
    assert os.path.exists(path_data)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    assert size_hdf5 > 0
    # convert
    convert_dataset_to_hdf5(list_file, path_data, path_save, size_hdf5, tag)

if __name__ == '__main__':
    main()