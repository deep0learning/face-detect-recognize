#!/bin/bash
#img test
#python reg_test.py --img-path1 /home/lxy/Downloads/DataSet/Face_reg/prison_crop_203/11176/video2_11762.jpg  --img-path2 /home/lxy/Downloads/DataSet/Face_reg/prison_crop_203/11176/video2_11762.jpg  \
 #  --cmd-type imgtest
#python reg_test.py --img-path1 /home/lxy/Downloads/DataSet/Face_reg/prison_crop_203/11176/video2_11784.jpg  --img-path2 /home/lxy/Downloads/DataSet/Face_reg/prison_crop_203/11176/video2_11762.jpg \
 #  --cmd-type imgtest
#python reg_test.py --img-path1 /home/lxy/Downloads/DataSet/Face_reg/prison_crop_203/11709/video1_10394.jpg  --img-path2 /home/lxy/Downloads/DataSet/Face_reg/prison_crop_203/11176/video2_11762.jpg  \
 #  --cmd-type imgtest
 #python reg_test.py --img-path1 /home/lxy/Downloads/DataSet/Face_reg/prison_result/prison_6244_v1/5859/v6244_20883.jpg  --img-path2 /home/lxy/Downloads/DataSet/Face_reg/id_236/id_241_crop/5859.jpg  \
  # --cmd-type imgtest

#compare file
#python reg_test.py --img-path1 ./c1.txt  --img-path2 ./c2.txt --cmd-type file_compare

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

##test train img
python reg_test.py --file-in ./output/id_256.txt  --data-file ./output/prison_6244.txt --id-dir /home/lxy/Downloads/DataSet/Face_reg/id_256/id_256_crop/ --base-dir /home/lxy/Downloads/DataSet/Face_reg/prison_img_test/prison_6244/ \
     --save-dir /home/lxy/Downloads/DataSet/Face_reg/prison_result/prison_6244/  --failed-dir  /home/lxy/Downloads/DataSet/Face_reg/prison_result/failed_dir  --label-file ./output/prison_label_256.pkl --cmd-type dbtest
#python reg_test.py --file-in ./output/id_256.txt  --data-file ./output/prison_0918.txt --id-dir /home/lxy/Downloads/DataSet/Face_reg/id_256/id_256_crop/ --base-dir /home/lxy/Downloads/DataSet/Face_reg/prison_img_test/prison_0918 \
 #   --save-dir /home/lxy/Downloads/DataSet/Face_reg/prison_result/prison_0918/  --failed-dir  /home/lxy/Downloads/DataSet/Face_reg/prison_result/failed_dir  --label-file ./output/prison_label_256.pkl --cmd-type dbtest
## reg form imglist
#python reg_test.py --file-in ./output/id_256.txt  --data-file ./output/prison_6244_list.txt --id-dir /home/lxy/Downloads/DataSet/Face_reg/id_256/id_256_crop/ \
 #       --base-dir /home/lxy/Downloads/DataSet/Face_reg/video_align/video6244/video_grab_align/  --save-dir /home/lxy/Downloads/DataSet/Face_reg/prison_result/prison_6244_v3  \
  #      --failed-dir  /home/lxy/Downloads/DataSet/Face_reg/prison_result/failed_6244dir_v3  --label-file ./output/prison_label_256.pkl --cmd-type dbtest