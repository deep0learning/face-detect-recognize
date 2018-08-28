# License
# requirests
    *device: GPU memory >= 2000M (my own is GTX1070), Cpu memory (my own is 16G)
    *deeplearning framework: (if you have mxnet and tensorflow models, install mxnet-cu80=1.20   tensorflow-gpu=1.20 pytorch=0.4 ) pycaffe 
    *python models: numpy scipy sklearn Easydict annoy pickle h5py opencv2 or opencv3
    *python2
# Contents
# Face_Detect and Face_recognize
    * The directories -12net 24net 48net  are designed for face detection training
    * demo  -- for face detect task (only for inference)
    * face_test -- for face recognition task ( only for inference)
    * video_demo -- face detection and face recognition merge together. The files will be used for recognize faces detected in the input video.
    * models -- saved face detect models and face recognize models
    * prepare_data -- generate datas for face detect training  when using caffe
# Demo
    *mtcnn_config.py -- including some parameters used to control the face detection. The details are given in it
    *test.py -- The top file, accepting the video input and image list file, detects faces and saves the faces to directory.
    *Detector.py --including face detector class API, there is opencv model and MTCNN model.
    *align.py -- including methods for align the detected faces
    *tools_matrix.py --deal with the output of Pnet, Rnet and Onet
    *run.sh -- used to run examples in command
# face_test
    *face_config.py --including some parameters used to control the face recognition. The details are given in it
    *reg_test.py --The top file, accepting the image list file and image pairs, recognizes the input faces and saves  these faces to directory.
                 -- "confuMat_test" the function used to generate distance datas for Confusion_Matrix.py input
    *face_model.py --including face recognition class API, there is caffe model,mxnet model, tensorflow model and pytorch model.
    *confusion_matrix.py --used to generate confusion_matrix
    *gen_file_test.py --A tool of script processing. 
    *test.sh --used to run examples in command
# video_demo
    *reg_demo.py --Merge the face detection with face recognition. The file will use the face detector class API in Detector.py and 
                 -- the face recognize class API in face_model.py. It accept video input.
    *run.sh --used to run examples in command