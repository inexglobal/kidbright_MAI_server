from flask import *
from fileinput import filename

import gdown
import urllib.request

import tuna

import threading, queue
import ctypes
import zipfile
import requests
from pathlib import Path

import sys, json, os, time, logging, random, shutil, tempfile, subprocess, re, platform, io
import base64
import numpy as np
import cv2
import utils.helper as helper
sys.path.append(".")


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

print("BACKEND : " + BACKEND)
print("DEVICE : " + DEVICE)
PROJECT_PATH = "./projects" if BACKEND == "COLAB" else "./projects"
PROJECT_FILENAME = "project.json"
PROJECT_ZIP = "project.zip"
OUTPUT_FOLDER = "output"
TEMP_FOLDER = "temp"

STAGE = 0 #0 none, 1 = prepare dataset, 2 = training, 3 = trained, 4 = converting, 5 converted

report_queue = queue.Queue()
train_task = None
report_task = None

#==================================== Server Configuration ====================================#
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
logging.basicConfig(level=logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

@app.route('/')
def index():
    return "Hello World"

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':        
        f = request.files['project']
        project_id = request.form['project_id']
        project_path = os.path.join(PROJECT_PATH, project_id)
        helper.create_not_exist(project_path)
        f.save(os.path.join(project_path, PROJECT_ZIP))
        
        return jsonify({'result': 'success'})

@app.route("/train", methods=["POST"])
def start_training():
    global train_task, report_queue
    print("start training process")
    data = request.get_json()
    project_id = data["project"]
    train_task = threading.Thread(target=training_task, args=(project_id,report_queue,))
    train_task.start()
    return jsonify({"result" : "OK"})

@app.route('/ping', methods=["GET","POST"])
def on_ping():
    return jsonify({"result":"pong"})

@app.route('/backend', methods=["GET"])
def on_backend():
    if request.method == 'GET':
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().strip()
        if "Jetson Nano" in model:
            return jsonify({"result":"OK",  "data" : "JETSON" })
        elif "Raspberry Pi" in model:
            return jsonify({"result":"OK",  "data" : "RPI" })
        elif "Nano" in model:
            return jsonify({"result":"OK",  "data" : "NANO" })
        else:
            return jsonify({"result":"OK",  "data" : "NONE" })

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

def training_task(project_id, q):
    global STAGE, current_model
    try:
        # 1 ========== prepare project ========= #
        STAGE = 1
        q.put({"time":time.time(), "event": "initial", "msg" : "Start training step 1 ... prepare dataset"})
        # unzip project
        project_zip = os.path.join(PROJECT_PATH, project_id, PROJECT_ZIP)
        project_folder = os.path.join(PROJECT_PATH, project_id)
        with zipfile.ZipFile(project_zip, 'r') as zip_ref:
            zip_ref.extractall(project_folder)
        os.remove(project_zip)
        # read project file
        project = helper.read_json_file(os.path.join(PROJECT_PATH, project_id, PROJECT_FILENAME))        
        q.put({"time":time.time(), "event": "initial", "msg" : "target project id : "+project_id})
        # 2 ========== prepare dataset ========= #
        STAGE = 2
        # execute script "!python train.py -d custom --cuda -v slim_yolo_v2 -hr -ms"
        q.put({"time":time.time(), "event": "initial", "msg" : "Start training step 2 ... training"})
        cmd = "python train.py -d custom --cuda -v slim_yolo_v2 -hr -ms"
        
        #subprocess.run(cmd, cwd="./", shell=True)
        subprocess.run(cmd, shell=True)
        # 3 ========== training ========= #
        
    finally:
        print("Thread ended")



@app.route("/terminate_training", methods=["POST"])
def terminate_training():
    global train_task, report_task, report_queue
    print("terminate current training process")
    if train_task and train_task.is_alive():
        kill_thread(train_task)
        print("send kill command")
    #if report_task and report_task.is_alive():
    #    report_queue.put({"time": time.time(), "event" : "terminate", "msg":"Training terminated"})
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
    len_arg = len(sys.argv)
    if len_arg > 2:
        if sys.argv[1] == "tuna" and sys.argv[2]:
            print("=== start tuna ===")
            tuna.run_tuna(5000,sys.argv[2])

    app.run(host="0.0.0.0",debug=True)
