# -*- coding: utf-8 -*-
from flask import *
from fileinput import filename

import urllib.request

import tuna

import threading, queue
import zipfile
import requests
from pathlib import Path

import sys, json, os, time, logging, random, shutil, tempfile, subprocess, re, platform, io
import base64
import numpy as np
import cv2
#---- helper ----#
from utils.message_announcer import MessageAnnouncer
import utils.helper as helper
sys.path.append(".")
#---- train ----#
from train_object_detection import train_object_detection
#---- converter ----#
from convert import torch_to_onnx, onnx_to_ncnn, gen_input
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import *

app = Flask(__name__)

#==================================== Define Variables ====================================#
UNAME = platform.uname()
BACKEND = ""
DEVICE = ""
if UNAME.system == "Windows":
    BACKEND = "EDGE"
    DEVICE = "WINDOWS"
elif 'COLAB_GPU' in os.environ:
    BACKEND = "COLAB"
    DEVICE = "COLAB"
else:
    with open("/proc/device-tree/model", "r") as f:
        model = f.read().strip()
        if "Jetson Nano" in model:
            DEVICE = "JETSON"
            BACKEND = "EDGE"
        elif "Raspberry Pi" in model:
            DEVICE = "RPI"
            BACKEND = "EDGE"
        elif "Nano" in model:
            DEVICE = "NANO"
            BACKEND = "EDGE"

PROJECT_PATH = "./projects" if BACKEND == "COLAB" else "./projects"
PROJECT_FILENAME = "project.json"
PROJECT_ZIP = "project.zip"
OUTPUT_FOLDER = "output"
TEMP_FOLDER = "temp"

STAGE = 0 #0 none, 1 = prepare dataset, 2 = training, 3 = trained, 4 = converting, 5 converted

reporter = MessageAnnouncer()
train_task = None
convert_task = None

#==================================== Server Configuration ====================================#
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
logging.basicConfig(level=logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

@app.route('/')
def index():
    return "Hello World"

@app.route('/listen', methods=['GET'])
def listen():
    def stream():
        messages = reporter.listen()  # returns a queue.Queue
        while True:
            msg = messages.get()  # blocks until a new message arrives
            yield msg

    return Response(stream(), mimetype='text/event-stream')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':        
        f = request.files['project']
        project_id = request.form['project_id']
        project_path = os.path.join(PROJECT_PATH, project_id)
        helper.recreate_folder(project_path)
        f.save(os.path.join(project_path, PROJECT_ZIP))
        
        return jsonify({'result': 'success'})

@app.route("/train", methods=["POST"])
def start_training():
    global train_task, reporter
    print("start training process")
    data = request.get_json()
    project_id = data["project"]
    train_task = threading.Thread(target=training_task, args=(project_id,reporter,))
    train_task.start()
    return jsonify({"result" : "OK"})

@app.route("/download", methods=["GET"])
def download_file():
    global convert_task, reporter
    # convert project
    project_id = request.args.get("project_id")
    if not project_id:
        return "Fail"
    convert_task = threading.Thread(target=convert_task, args=(project_id,reporter,))
    convert_task.start()
    return jsonify({"result" : "OK"})
    

@app.route('/ping', methods=["GET","POST"])
def on_ping():
    return jsonify({"result":"pong"})

@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Methods']='*'
    response.headers['Access-Control-Allow-Origin']='*'
    response.headers['Vary']='Origin'
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, Access-Control-Allow-Headers, X-Requested-With"
    return response

@app.route('/projects/<path:path>')
def send_report(path):
    return send_from_directory('projects', path)

def convert_task(project_id, q):
    global STAGE
    project_path = os.path.join(PROJECT_PATH, project_id)
    best_ap_file = os.path.join(project_path, "output", "best_map.pth")
    if not os.path.exists(best_ap_file):
        return jsonify({"result" : "FAIL", "reason":"No best_map.pth file"})
    device = torch.device("cpu")

    project = helper.read_json_file(os.path.join(project_path, PROJECT_FILENAME))
    project_model = project["trainConfig"]["modelType"]
    input_size = [416 , 416]
    model_label = [l["label"] for l in project["labels"]]

    if project_model == "slim_yolo_v2":
        from models.slim_yolo_v2 import SlimYOLOv2 
        anchor_size = ANCHOR_SIZE
        num_classes = len(model_label)
        detect_threshold = float(project["trainConfig"]["objectThreshold"])
        iou_threshold = float(project["trainConfig"]["iouThreshold"])
        net = SlimYOLOv2(device, input_size=input_size, num_classes=num_classes, conf_thresh=detect_threshold, nms_thresh=iou_threshold, anchor_size=anchor_size)
    
    net.load_state_dict(torch.load(best_ap_file, map_location=device))
    net.to(device).eval()
    print('Finished loading model!')

    # convert to onnx and ncnn
    from torchsummary import summary
    summary(net.to("cpu"), input_size=(3, input_size[0], input_size[1]), device="cpu")

    # convert model    
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

    cmd2 = "tools/spnntools calibrate -p=out/opt.param -b=out/opt.bin -i=./data/custom/test_images -o=out/opt.table --m=127.5,127.5.0,127.5.0 --n=1.0,1.0,1.0 --size=224,224 -c -t=4"
    os.system(cmd2)

    cmd3 = "tools/spnntools quantize out/opt.param out/opt.bin out/opt_int8.param out/opt_int8.bin out/opt.table"
    os.system(cmd3)


    # convert pth to onnx
    # onnx_out = os.path.join(project_path, "output", "model.onnx")
    # ncnn_out_param = os.path.join(project_path, "output", "model.param")
    # ncnn_out_bin = os.path.join(project_path, "output", "model.bin")
    # input_shape = (3, 416, 416)
    # with torch.no_grad():
    #     torch_to_onnx(net.to("cpu"), input_shape, onnx_out, device="cpu")
    #     onnx_to_ncnn(input_shape, onnx=onnx_out, ncnn_param=ncnn_out_param, ncnn_bin=ncnn_out_bin)
        
    # # encode mode with spnntools

    # os.environ['PKG_CONFIG_PATH'] = ':/root/opencv-3.4.13/lib/pkgconfig'
    # os.environ['LD_LIBRARY_PATH'] += ':/root/opencv-3.4.13/lib'
    # os.environ['PATH'] += ':/root/opencv-3.4.13/bin'
    # output_model_optimize_bin_path = os.path.join(project_path, "output", "model_opt.bin")
    # output_model_optimize_param_path = os.path.join(project_path, "output", "model_opt.param")
    # cmd = f"tools/spnntools optimize {ncnn_out_param} {ncnn_out_bin} {output_model_optimize_param_path} {output_model_optimize_bin_path}"
    # os.system(cmd)
    # #spnntools calibrate -p=opt.param -b=opt.bin -i=/content/images -o=opt.table --m=127.5,127.5.0,127.5.0 --n=1.0,1.0,1.0 --size=224,224 -c -t=4
    # output_model_calibrate_table = os.path.join(project_path, "output", "model_opt.table")
    # test_images_path = os.path.join(project_path, "datasets", "JPEGImages")
    # cmd2 = f"tools/spnntools calibrate -p={output_model_optimize_param_path} -b={output_model_optimize_bin_path} -i={test_images_path} -o={output_model_calibrate_table} --m=127.5,127.5.0,127.5.0 --n=1.0,1.0,1.0 --size=224,224 -c -t=4"
    # os.system(cmd2)
    # # quantize model
    # # spnntools quantize opt.param opt.bin opt_int8.param opt_int8.bin opt.table
    # output_model_quantize_bin_path = os.path.join(project_path, "output", "model_int8.bin")
    # output_model_quantize_param_path = os.path.join(project_path, "output", "model_int8.param")
    # cmd3 = f"tools/spnntools quantize {output_model_optimize_param_path} {output_model_optimize_bin_path} {output_model_quantize_param_path} {output_model_quantize_bin_path} {output_model_calibrate_table}"
    # os.system(cmd3)
    

def training_task(project_id, q):
    global STAGE, current_model
    try:
        # 1 ========== prepare project ========= #
        STAGE = 1
        # for i in range(50):
        #     q.announce({"time":time.time(), "event": "initial", "msg" : "Start training step 1 ... prepare dataset"})
        #     time.sleep(1)
        q.announce({"time":time.time(), "event": "initial", "msg" : "Start training step 1 ... prepare dataset"})
        
        # unzip project        
        project_zip = os.path.join(PROJECT_PATH, project_id, PROJECT_ZIP)
        project_folder = os.path.join(PROJECT_PATH, project_id)
        with zipfile.ZipFile(project_zip, 'r') as zip_ref:
            zip_ref.extractall(project_folder)
        os.remove(project_zip)
        # read project file
        project = helper.read_json_file(os.path.join(PROJECT_PATH, project_id, PROJECT_FILENAME))        
        q.announce({"time":time.time(), "event": "initial", "msg" : "target project id : "+project_id})
        # 2 ========== prepare dataset ========= #
        STAGE = 2
        # execute script "!python train.py -d custom --cuda -v slim_yolo_v2 -hr -ms"
        q.announce({"time":time.time(), "event": "initial", "msg" : "Start training step 2 ... training"})
        
        output_path = os.path.join(project_folder, "output")
        dataset_path = os.path.join(project_folder, "datasets")
        #{'validateMatrix': 'validation-accuracy', 'saveMethod': 'Best value after n epoch', 'modelType': 'Resnet18', 'weights': 'resnet18', 'inputWidth': 320, 'inputHeight': 240, 'train_split': 80, 'epochs': 100, 'batch_size': 32, 'learning_rate': 0.001}
        # check if project has trainConfig and it valid        

        # label format in json lables : [ {label: "label1"}, {label: "label2"}]
        model_label = [l["label"] for l in project["labels"]]
        train_object_detection(project, output_path, project_folder,q,
            high_resolution=True, 
            multi_scale=True, 
            cuda=False, 
            learning_rate=project["trainConfig"]["learning_rate"], 
            batch_size=project["trainConfig"]["batch_size"],
            start_epoch=0, 
            epoch=project["trainConfig"]["epochs"],
            train_split=project["trainConfig"]["train_split"],
            model_type=project["trainConfig"]["modelType"],
            model_weight=None,
            validate_matrix=project["trainConfig"]["validateMatrix"],
            save_method=project["trainConfig"]["saveMethod"],
            step_lr=(150, 200),
            labels=model_label,
            momentum=0.9,
            weight_decay=5e-4,
            warm_up_epoch=6
        )
        
        # 3 ========== training ========= #
        
    finally:
        print("Thread ended")



@app.route("/terminate_training", methods=["POST"])
def terminate_training():
    global train_task, reporter
    print("terminate current training process")
    if train_task and train_task.is_alive():
        kill_thread(train_task)
        print("send kill command")
    #if report_task and report_task.is_alive():
    #    reporter.announce({"time": time.time(), "event" : "terminate", "msg":"Training terminated"})
    time.sleep(3)
    return jsonify({"result" : "OK"})

@app.route("/convert_model", methods=["POST"])
def handle_convert_model():
    print("convert model")
    data = request.get_json()
    res = {}
    project_id = data["project_id"]
    project_backend = None
    if "backend" in data:
        project_backend = data["backend"]
    if not project_id:
        return "Fail"
    output_path = os.path.join(PROJECT_PATH, project_id, "output")
    files = [os.path.join(output_path,f) for f in os.listdir(output_path) if f.endswith(".h5")]
    if len(files) <= 0:
        return "Fail"
    
    project_file = os.path.join(PROJECT_PATH, project_id, PROJECT_FILENAME)
    project = helper.read_json_file(project_file)
    model = project["project"]["project"]["model"]
    cmd_code = model["code"]
    config = helper.parse_json(cmd_code)
    raw_dataset_path = os.path.join(PROJECT_PATH, project_id, RAW_DATASET_FOLDER)
    output_model_path = os.path.join(PROJECT_PATH, project_id, OUTPUT_FOLDER)
    tfjs_model_path = os.path.join(output_model_path,"tfjs")
    
    #--- tfjs converter ---#
    convert_res = subprocess.run(["tensorflowjs_converter --input_format keras "+files[0] + " " + tfjs_model_path], stdout=subprocess.PIPE, shell=True)
    subprocess.run(["sed -i 's/LecunNormal/RandomNormal/g' "+tfjs_model_path+"/model.json"], shell=True)
    subprocess.run(["sed -i 's/Functional/Model/g' "+tfjs_model_path+"/model.json"], shell=True)
    #--- edge converter ---#
    if project_backend == "RPI" or project_backend == "NANO":
        converter = Converter("edgetpu", normalize, raw_dataset_path)
        converter.convert_model(files[0])
    elif not project_backend or project_backend == "JETSON" :
        converter = Converter("tflite_dynamic", normalize, raw_dataset_path)
        converter.convert_model(files[0])
        src_name = os.path.basename(files[0]).split(".")
        src_path = os.path.dirname(files[0])
        src_tflite = os.path.join(src_path,src_name[0] + ".tflite")
        des_tflite = os.path.join(src_path,src_name[0] + "_edgetpu.tflite")
        shutil.copyfile(src_tflite, des_tflite)
    shutil.make_archive(os.path.join(PROJECT_PATH, project_id, "model"), 'zip', output_model_path)

    return jsonify({"result" : "OK"})

@app.route("/download_model", methods=["GET"])
def handle_download_model():
    print("download model file")
    project_id = request.args.get("project_id")
    model_export = os.path.join(PROJECT_PATH,project_id,"model.zip")
    return send_file(model_export, as_attachment=True)

@app.route("/inference_image", methods=["POST"])
def handle_inference_model():
    global STAGE, current_model
    if 'image' not in request.files:
        return "No image"
    if STAGE < 3:
        return "Training not success yet :" + str(STAGE)
    
    tmp_img = request.files['image']
    project_id = request.form['project_id']
    model_type = request.form['type']

    if not tmp_img:
        return "Image null or something"
    
    target_file_path = os.path.join(PROJECT_PATH, project_id, TEMP_FOLDER)
    helper.create_not_exist(target_file_path) 
    target_file = os.path.join(target_file_path, tmp_img.filename)
    tmp_img.save(target_file)    

    if model_type == "classification":
        orig_image, img = helper.prepare_image(target_file, current_model, current_model.input_size)
        elapsed_ms, prob, prediction = current_model.predict(img)
        return jsonify({"result" : "OK","prediction":prediction, "prob":np.float64(prob)})
    elif model_type == "detection":
        threshold = float(request.form['threshold'])
        orig_image, input_image = helper.prepare_image(target_file, current_model, current_model._input_size)
        height, width = orig_image.shape[:2]
        prediction_time, boxes, probs = current_model.predict(input_image, height, width, threshold)
        labels = current_model._labels
        bboxes = []
        for box, classes in zip(boxes, probs):
            x1, y1, x2, y2 = box
            bboxes.append({
                "x1" : np.float64(x1), 
                "y1" : np.float64(y1), 
                "x2" : np.float64(x2), 
                "y2" : np.float64(y2), 
                "prob" : np.float64(classes.max()), 
                "label" : labels[np.argmax(classes)]
            })
        return jsonify({"result" : "OK", "boxes": bboxes})
    else:
        return jsonify({"result" : "FAIL","reason":"model type not specify"})


if __name__ == '__main__':
    print("BACKEND : " + BACKEND)
    print("DEVICE : " + DEVICE)
    len_arg = len(sys.argv)
    if len_arg > 2:
        if sys.argv[1] == "tuna" and sys.argv[2]:
            print("=== start tuna ===")
            tuna.run_tuna(5000,sys.argv[2])

    app.run(host="0.0.0.0",debug=True)
