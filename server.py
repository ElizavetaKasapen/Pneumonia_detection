from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

# it works!
@app.route('/upload_image/', methods=['POST'])
def upload_image():
  print ('I got clicked!')
  if request.method == 'POST':
        # we will get the file from the request
        file = request.files['upload_image']
        # convert that to bytes
        #img_bytes = file.read()
  return '<h1>Click.</h1>'

if __name__ == '__main__':
  app.run(debug=True)
