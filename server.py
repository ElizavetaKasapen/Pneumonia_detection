from flask import Flask, render_template, request, flash, redirect
import Pneumonia_detection_net
from PIL import Image
import io
import base64

app = Flask(__name__)
net = Pneumonia_detection_net.nn_for_pneumonia_detection()

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
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes))
    image = image.convert("RGB")
    image.save(image, "JPEG")
    encoded_img_data = base64.b64encode(image.getvalue())
    input_img = encoded_img_data.decode('utf-8')
    prediction = net.to_predict(img_bytes)
  return render_template('index.html', filename = file.filename, image = input_img, prob=prediction)

if __name__ == '__main__':
  app.run(debug=True)
