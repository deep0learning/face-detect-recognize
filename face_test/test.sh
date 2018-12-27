#!/bin/bash
#img test
#python reg_test.py --img-path1 /home/lxy/Develop/faster_rcnn/Mask_RCNN/8973_0.jpg  \
#               --img-path2 /home/lxy/Downloads/DataSet/Face_reg/prison_video_result/prison_3038/8973/8973_0.jpg --cmd-type imgtest
 #python reg_test.py --img-path1 /home/lxy/Downloads/DataSet/Face_reg/prison_video_result/prison_3038/3975/3975.jpg \
  #                  --img-path2 /home/lxy/Develop/faster_rcnn/Mask_RCNN/images/3975_20.jpg --cmd-type imgtest
#python reg_test.py --img-path1 /home/lxy/Develop/faster_rcnn/Mask_RCNN/images/3975_20.jpg \
 #        --img-path2 /home/lxy/Develop/faster_rcnn/Mask_RCNN/images/8973.jpg  --cmd-type imgtest
#python reg_test.py --img-path1 /home/lxy/Desktop/sph_wrong/2.jpg \
 #        --img-path2 /home/lxy/Desktop/sph_wrong/2_n.jpg  --cmd-type imgtest
#python reg_test.py --img-path1 /home/lxy/Desktop/wrong/1.jpeg \
 #        --img-path2 /home/lxy/Desktop/wrong/1-0.95.jpeg  --cmd-type imgtest
#python reg_test.py --img-path1 /home/lxy/Develop/Center_Loss/mtcnn-caffe/image/pics/11.jpeg \
 #        --img-path2 /home/lxy/Develop/Center_Loss/mtcnn-caffe/image/pics/2.jpeg  --cmd-type imgtest
 #../image/5-0.484.jpeg \
#python reg_test.py --img-path1  ../image/6-0.10.jpeg \
 #        --img-path2 ../image/66.jpeg  --cmd-type imgtest
 #python reg_test.py --img-path1 /home/lxy/Downloads/DataSet/Face_reg/Downloads/44.jpeg \
  #       --img-path2 /home/lxy/Downloads/DataSet/Face_reg/Downloads/2a.jpeg  --cmd-type imgtest
#compare file
#python reg_test.py --img-path1 /home/lxy/Downloads/feat11.txt  --img-path2 /home/lxy/Downloads/feat22.txt --cmd-type comparefile

# db test
#python  reg_test.py --file-in ./output/id_241.txt  --data-file ./output/prison_203.txt --id-dir /home/lxy/Downloads/DataSet/Face_reg/id_236/id_241_crop/ --base-dir /home/lxy/Downloads/DataSet/Face_reg/prison_crop_203/ \
 #     --save-dir /home/lxy/Downloads/DataSet/Face_reg/train/ --label-file ./output/prison_241.pkl
#python reg_test.py --file-in ./output/id_241.txt  --data-file ./output/prison_all.txt --id-dir /home/lxy/Downloads/DataSet/Face_reg/id_236/id_241_crop/ \
 #   --base-dir /home/lxy/Downloads/DataSet/Face_reg/prison_crop_100/align/all/  --save-dir /home/lxy/Downloads/DataSet/Face_reg/train/ --label-file ./output/prison_241.pkl --cmd-type dbtest

# test model db

##test 3 videos
#python reg_test.py --file-in ./output/id_241.txt  --data-file ./output/prison_3038.txt --id-dir /home/lxy/Downloads/DataSet/Face_reg/id_236/id_241_crop/ \
 #   --base-dir /home/lxy/Downloads/DataSet/Face_reg/prison_3038/face_align/  --save-dir /home/lxy/Downloads/DataSet/Face_reg/prison_result/prison_3038/ --label-file ./output/prison_241.pkl --cmd-type dbtest
#video4907
#python reg_test.py --file-in ./output/id_241.txt  --data-file ./output/prison_4907.txt --id-dir /home/lxy/Downloads/DataSet/Face_reg/id_236/id_241_crop/ \
#    --base-dir /home/lxy/Downloads/DataSet/Face_reg/prison_4907/  --save-dir /home/lxy/Downloads/DataSet/Face_reg/prison_result/prison_4907/ --label-file ./output/prison_241.pkl --cmd-type dbtest
#video 1737
#python reg_test.py --file-in ./output/id_241.txt  --data-file ./output/prison_1737.txt --id-dir /home/lxy/Downloads/DataSet/Face_reg/id_236/id_241_crop/ \
 #   --base-dir /home/lxy/Downloads/DataSet/Face_reg/prison_1737/  --save-dir /home/lxy/Downloads/DataSet/Face_reg/prison_result/prison_1737/ --label-file ./output/prison_241.pkl --cmd-type dbtest

##multi model test train img
#python reg_test.py --file-in ./output/id_256.txt  --data-file ./output/prison_6244.txt --id-dir /home/lxy/Downloads/DataSet/Face_reg/id_256/id_256_crop/ --base-dir /home/lxy/Downloads/DataSet/Face_reg/prison_img_test/prison_6244/ \
 #    --save-dir /home/lxy/Downloads/DataSet/Face_reg/prison_result/prison_6244/  --failed-dir  /home/lxy/Downloads/DataSet/Face_reg/prison_result/failed_dir  --label-file ./output/prison_label_256.pkl --cmd-type dbtest \
  #   --top2-failed-dir  /home/lxy/Downloads/DataSet/Face_reg/prison_result/top2_failed_dir
#python multi_model.py --file-in ./output/id_256.txt  --data-file ./output/prison_0918.txt --id-dir /home/lxy/Downloads/DataSet/Face_reg/id_256/id_256_crop/ --base-dir /home/lxy/Downloads/DataSet/Face_reg/prison_img_test/prison_0918 \
 #  --save-dir /home/lxy/Downloads/DataSet/Face_reg/prison_result/prison_0918/  --failed-dir  /home/lxy/Downloads/DataSet/Face_reg/prison_result/failed_dir  --label-file ./output/prison_label_256.pkl --cmd-type dbtest
## reg form imglist
#python reg_test.py --file-in ./output/id_256.txt  --data-file ./output/img1.txt --id-dir /home/lxy/Downloads/DataSet/Face_reg/id_256/id_256_crop/  --base-dir /home/lxy/Downloads/DataSet/Face_reg/prison_img_test/prison_6244/ \
 #      --save-dir /home/lxy/Downloads/DataSet/Face_reg/prison_result/prison_6244/  --failed-dir  /home/lxy/Downloads/DataSet/Face_reg/prison_result/failed_dir  --label-file ./output/prison_label_256.pkl --cmd-type dbtest
### generate test images
#python reg_test.py --file-in ./output/id_256.txt  --data-file ./output/prison_failed.txt --id-dir /home/lxy/Downloads/DataSet/Face_reg/id_256/id_256_crop/ --base-dir /home/lxy/Downloads/DataSet/Face_reg/prison_result/top_failed_dir/ \
 #    --save-dir /home/lxy/Downloads/DataSet/Face_reg/prison_result/prison_2/  --failed-dir  /home/lxy/Downloads/DataSet/Face_reg/prison_result/failed_dir  --label-file ./output/prison_label_256.pkl --cmd-type dbtest \
  #   --top2-failed-dir  /home/lxy/Downloads/DataSet/Face_reg/prison_result/top2_failed_dir

## test highway data using test class
#python test_db.py --file-in ./output/id_50000.txt  --data-file ./output/photo_80000.txt --id-dir /home/lxy/Downloads/DataSet/Face_reg/HighRailway/id_50000_crop --base-dir /home/lxy/Downloads/DataSet/Face_reg/HighRailway/photo_crop/ \
 #   --save-dir /home/lxy/Downloads/DataSet/Face_reg/highway_result/v1/  --failed-dir  /home/lxy/Downloads/DataSet/Face_reg/highway_result/failed_dir  --label-file ./output/label_50000.pkl --cmd-type dbtest \
  #   --top2-failed-dir  /home/lxy/Downloads/DataSet/Face_reg/highway_result/top2_failed_dir 
python test_db.py --file-in ./output/id_52611.txt  --data-file ./output/prison_6244.txt --id-dir /home/lxy/Downloads/DataSet/Face_reg/HighRailway/id_52611_crop/id_50000_crop \
                  --base-dir /home/lxy/Downloads/DataSet/Face_reg/prison_img_test/prison_6244 --save-dir /home/lxy/Downloads/DataSet/Face_reg/highway_result/v2/  \
                  --failed-dir  /home/lxy/Downloads/DataSet/Face_reg/highway_result/failed_dir  --label-file ./output/label_52611.pkl --cmd-type dbtest \
                  --top2-failed-dir  /home/lxy/Downloads/DataSet/Face_reg/highway_result/top2_failed_dir