#from vide
#python test.py --file-in /home/lxy/Downloads/DataSet/videos/prison_video/n_716/20180711141737.ts --base-name v1737 --save-dir /home/lxy/Downloads/DataSet/Face_reg/video_align/video1737/video_grab_align/ \
 #       --save-dir2 /home/lxy/Downloads/DataSet/Face_reg/video_align/video1737/video_grab_widen/ --cmd-type video
 #python test.py --file-in /home/lxy/Downloads/DataSet/videos/prison_video/20180828.mp4 --base-name v0828 --save-dir /home/lxy/Downloads/DataSet/Face_reg/video_align/video0828/video_grab_align/ \
  #     --save-dir2 /home/lxy/Downloads/DataSet/Face_reg/video_align/video0828/video_grab_widen/ --cmd-type video
#txtfile
#python test.py --file-in ../face_test/output/id_256.txt --base-dir /home/lxy/Downloads/DataSet/Face_reg/id_256/id/ --save-dir /home/lxy/Downloads/DataSet/Face_reg/id_256/id_256_crop/  --img-size 112,112 --cmd-type txtfile
#python test.py --file-in ../face_test/output/photo_80000.txt --base-dir /home/lxy/Downloads/DataSet/Face_reg/HighRailway/photo/ --save-dir /home/lxy/Downloads/DataSet/Face_reg/HighRailway/photo_crop/  --img-size 112,112 --cmd-type txtfile
#python test.py --file-in ./output/failed_face.txt --base-dir /home/lxy/Downloads/DataSet/Face_reg/HighRailway/photo/ --save-dir /home/lxy/Downloads/DataSet/Face_reg/HighRailway/photo_crop/  --img-size 112,112 --cmd-type txtfile
python test.py --file-in ./output/failed_face2.txt --base-dir /home/lxy/Downloads/DataSet/Face_reg/HighRailway/id/ --save-dir /home/lxy/Downloads/DataSet/Face_reg/HighRailway/id_50000_crop/  --img-size 112,112 --cmd-type txtfile
#new videos
#python test.py --file-in /home/lxy/Downloads/DataSet/videos/down_2018-08-18/10.143.113.129-4-20180814142002-20180814142559-687508324.ts --base-name v8324 --save-dir /home/lxy/Downloads/DataSet/Face_reg/video_align/video8324/video_grab_align/ \
 #       --save-dir2 /home/lxy/Downloads/DataSet/Face_reg/video_align/video8324/video_grab_widen/ --cmd-type video
#816
###here to reg
#python test.py --file-in /home/lxy/Downloads/DataSet/videos/down_2018-08-18/10.143.113.129-1-20180816114311-20180816114729-687031616.ts --base-name v1616 --save-dir /home/lxy/Downloads/DataSet/Face_reg/video_align/video1616/video_grab_align/ \
 #       --save-dir2 /home/lxy/Downloads/DataSet/Face_reg/video_align/video1616/video_grab_widen/ --cmd-type video


#img test
#python test.py --img-path1 /home/lxy/Downloads/DataSet/Face_reg/HighRailway/photo/511301195501302294_02.jpg  --min-size 20 --cmd-type imgtest
#show form txtfile
#python test.py --file-in ./output/failed_face2.txt  --min-size 15 --cmd-type showtxt
#camera test
#python test.py --cmd-type camera