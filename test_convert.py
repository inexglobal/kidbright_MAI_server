import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import *
import torch.utils.data as data
import numpy as np
import cv2
import tools
import time
import os.path as osp

os.environ['PKG_CONFIG_PATH'] = ':/root/opencv-3.4.13/lib/pkgconfig'
os.environ['LD_LIBRARY_PATH'] += ':/root/opencv-3.4.13/lib'
os.environ['PATH'] += ':/root/opencv-3.4.13/bin'

device = torch.device("cpu")

input_size = [416, 416]

print('test on custom ...')
CUSTOM_CLASSES = ['face']
num_classes = len(CUSTOM_CLASSES)
conf_thresh = 0.1
nms_thresh = 0.5
trained_model= "out/best_map.pth"

from models.slim_yolo_v2 import SlimYOLOv2 

net = SlimYOLOv2(device, 
  input_size=input_size, 
  num_classes=num_classes, 
  conf_thresh=conf_thresh, 
  nms_thresh=nms_thresh, 
  anchor_size=ANCHOR_SIZE
)

net.load_state_dict(torch.load(trained_model, map_location=device))
net.to(device).eval()
print('Finished loading model!')

# convert to onnx and ncnn
from torchsummary import summary
summary(net.to("cpu"), input_size=(3, input_size[0], input_size[1]), device="cpu")

print("export model")
net.no_post_process = True
from convert import *
onnx_out="out/yolov2.onnx"
ncnn_out_param = "out/yolov2.param"
ncnn_out_bin = "out/yolov2.bin"
input_shape = (3, input_size[0], input_size[1])
import os
if not os.path.exists("out"):
    os.makedirs("out")
with torch.no_grad():
    torch_to_onnx(net.to("cpu"), input_shape, onnx_out, device="cpu")
    onnx_to_ncnn(input_shape, onnx=onnx_out, ncnn_param=ncnn_out_param, ncnn_bin=ncnn_out_bin)
    print("convert end, ctrl-c to exit")
net.no_post_process = False

cmd = "tools/spnntools optimize out/yolov2.param out/yolov2.bin out/opt.param out/opt.bin"
os.system(cmd)

cmd2 = "tools/spnntools calibrate -p=out/opt.param -b=out/opt.bin -i=./data/custom/test_images -o=out/opt.table --m=127.5,127.5,127.5 --n=1.0,1.0,1.0 --size=224,224 -c -t=4"
os.system(cmd2)

cmd3 = "tools/spnntools quantize out/opt.param out/opt.bin out/opt_int8.param out/opt_int8.bin out/opt.table"
os.system(cmd3)


#awnntools optimize model.param model.bin opt.param opt.bin 
#awnntools calibrate -p="opt.param" -b="opt.bin" -i="/images" -m="127.5,127.5,127.5" -n="0.0078125, 0.0078125, 0.0078125" -o="opt.table" -s="224, 224"  -c -t=8 
#awnntools quantize opt.param opt.bin model_awnn.param model_awnn.bin opt.table