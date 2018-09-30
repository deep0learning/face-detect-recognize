from easydict import EasyDict

config = EasyDict()
#save the boxes widen images, 1: widen; 0: not
config.box_widen = 1
# after detection if widen boxes
config.det_box_widen = 0