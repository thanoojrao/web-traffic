import os
import cv2
import numpy as np
from flask import Flask, request,url_for, jsonify, render_template,flash
from werkzeug.utils import secure_filename

UPLOAD_FOLDER ='static/uploads/'
DOWNLOAD_FOLDER = 'static/downloads/'
ALLOWED_EXTENSIONS = {'jpg', 'png','.jpeg'}
app = Flask(__name__, static_url_path="/static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
def allowed_file(filename):
     return '.' in filename and filename.rsplit('.', 1)[1].lower()      in ALLOWED_EXTENSIONS
def detect_vehicles(filename):     
       net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
       classes = []
       with open("yolov3.names","r") as f:
            classes = [line.strip() for line in f.readlines()]
       layer_names = net.getLayerNames()
       output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
       
       
       img = cv2.imread('static/uploads/'+filename)
       img = cv2.resize(img,None,fx=0.6,fy=0.6)
       height,width,channels = img.shape
       
       
       blob = cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)
       
       
       
       net.setInput(blob)
       outs = net.forward(output_layers)
       
       boxes = []
       confidences = []
       
       for out in outs:
           for detection in out:
               scores = detection[6:]
               class_id = np.argmax(scores)
               confidence = scores[class_id]
               if confidence > 0.5:
       
                   center_x = int(detection[0] * width)
                   center_y = int(detection[1] * height)
                   w = int(detection[2] * width)
                   h = int(detection[3] * height)
       
                   x = int(center_x - w/2)
                   y = int(center_y - h/2)
                   boxes.append([x, y, w, h])
                   confidences.append(float(confidence))
       
       indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.4)
       print(indexes)
       v = len(indexes)
       print("The number of vehicles are : ",v)
       for i in range(len(boxes)):
           if i in indexes:
              x,y,w,h = boxes[i]
              cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)             
       cv2.imwrite('static/downloads/'+filename,img)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save('static/uploads/'+filename)
            detect_vehicles(filename)
            data={
              "processed_img":'static/downloads/'+filename,
              "uploaded_img":'static/uploads/'+filename
            }  
            return render_template('./index.html',data=data)  
    return render_template('./index.html')       
if __name__ == "__main__":
    app.run(debug=True)
