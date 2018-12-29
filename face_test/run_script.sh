#!/bin/bash

##for histgram
#python gen_file_test.py --file-in ./output/distance_top1.txt --out-file ./output/top1_hist_result2.txt --cmd-type hist
#python gen_file_test.py --file-in ./output/distance_top1.txt --out-file ./output/top1_hist_mx_result1.txt --cmd-type hist
#for top2 histgram
#python gen_file_test.py --file-in ./output/distance_top2.txt --out-file ./output/top2_hist_result2.txt --hist-max 0.3 --cmd-type hist
#python gen_file_test.py --file-in ./output/distance_top2.txt --out-file ./output/top2_hist_mx_result1.txt --hist-max 0.5 --cmd-type hist

##generate image list
#python gen_file_test.py --img-dir /home/lxy/Downloads/DataSet/Face_reg/prison_result/top_failed_dir  --out-file ./output/prison_failed.txt --cmd-type gen_filepath_2dir
#python gen_file_test.py --img-dir /home/lxy/Downloads/DataSet/Face_reg/HighRailway/photo_crop/  --out-file ./output/photo_80000.txt --cmd-type gen_filepath_1dir
#python gen_file_test.py --img-dir /home/lxy/Downloads/DataSet/Face_reg/HighRailway/id_50000_crop --out-file ./output/id_50000.txt --cmd-type gen_filepath_1dir
#python gen_file_test.py --img-dir /home/lxy/Downloads/DataSet/Face_reg/HighRailway/id_52611_crop/id_50000_crop --out-file ./output/id_52611.txt --cmd-type gen_filepath_1dir
python gen_file_test.py --img-dir /home/lxy/Downloads/DataSet/Face_reg/prison_img_test/prison_6244 --out-file ./output/prison_6244.txt --cmd-type gen_filepath_2dir
#python gen_file_test.py --img-dir /home/lxy/Downloads/DataSet/Face_reg/id_256/id_256_crop --out-file ./output/id_256.txt --cmd-type gen_filepath_1dir

##test image enhance
#python gen_file_test.py --img-dir /home/lxy/Downloads/DataSet/Face_reg/prison_img_test/prison_8324/111088/v8324_13043.jpg   --cmd-type imgenhance
#python gen_file_test.py --img-dir /home/lxy/Downloads/DataSet/Face_reg/prison_img_test/prison_8324/111088/v8324_13133.jpg   --cmd-type imgenhance
#python gen_file_test.py --img-dir /home/lxy/Downloads/DataSet/Face_reg/train/6384/6384_0.jpg   --cmd-type imgenhance

##generate pkl 
#python gen_file_test.py --img-dir /home/lxy/Downloads/DataSet/Face_reg/HighRailway/id_50000_crop --out-file ./output/label_50000.pkl --cmd-type gen_label_pkl
#python gen_file_test.py --img-dir /home/lxy/Downloads/DataSet/Face_reg/HighRailway/id_52611_crop/id_50000_crop  --out-file ./output/label_52611.pkl --cmd-type gen_label_pkl

### compare 2dirs
#python gen_file_test.py --img-dir /home/lxy/Downloads/DataSet/Face_reg/prison_result/prison_6244_v2/ --dir2-in /home/lxy/Downloads/DataSet/Face_reg/prison_result/server_0_6244/ \
 #       --save-dir /home/lxy/Downloads/DataSet/Face_reg/prison_result/prison_server_not/ --cmd-type compare_dir

## run histgram
#static distance vs confidence
#python histogram.py --key-name  distance --path2 ./output/distance_top2.csv   --command static2
#python histogram.py --path2 ./output/distance_top2.csv   --command tpr
#python histogram.py --path2 ./output/distance_top2.csv --label-file ./output/label_52611.pkl --top1-thres 1.0 --conf-thres 0.15  \
 #                 --save-dir /home/lxy/Downloads/DataSet/Face_reg/highway_result/static/ --command recog
python histogram.py --path1 ./output/prison_6244.txt  --out-file ./output/static_prison6244.txt --command data_static

## merge 2 txt file
#python gen_file_test.py --file-in ./output/id_256.txt --file2-in ./output/id_50000.txt --out-file ./output/id_10256.txt --cmd-type merge