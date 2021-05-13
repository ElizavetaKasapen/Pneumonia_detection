from flask import Flask, render_template, request, flash, redirect, send_from_directory
import Pneumonia_detection_net
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
net = Pneumonia_detection_net.nn_for_pneumonia_detection()
app.config["IMAGE_UPLOADS"] = "Uploads/"

@app.route('/')
def index():
  return render_template('index.html')

# it works!
@app.route('/upload_image/', methods=['POST'])
def upload_image():
  file = None
  prediction = None
  if request.method == 'POST':
    if 'upload_image' not in request.files:
      return render_template('index.html')
    # we will get the file from the request
    file = request.files['upload_image'] 
    if file.filename =="":
      return render_template('index.html')
    # convert that to bytes
    img_bytes = file.read()#for net
    prediction = net.to_predict(img_bytes)

    data = base64.b64encode(img_bytes)
    data = data.decode()
    img = '<img src="data:image/*;base64,{}">'.format(data)
    #image = Image.open(file)
    #image = image.convert("RGB")
    #image = image.save(file.filename)
    #encoded_img_data = base64.b64encode(image.getvalue())
    #encoded_img_data = encoded_img_data.decode('utf-8')
    ##encoded_img_data = base64.b64encode(image.getvalue())
    #input_img = encoded_img_data.decode('utf-8')
    file.save(os.path.join(app.config["IMAGE_UPLOADS"], file.filename))
  return render_template('index.html', filename = file.filename, prob=prediction, image = '')

@app.route('/uploads/<filename>')
def send_uploaded_file(filename=''):
    return send_from_directory(app.config["IMAGE_UPLOADS"], filename)

if __name__ == '__main__':
  app.run(debug=True)
