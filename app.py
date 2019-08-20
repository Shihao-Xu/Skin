from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time

from datetime import timedelta
import sys
sys.path.append('/home/JingyuXu/')
#import predict as pt
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import keras

img_path='/home/JingyuXu/skin/static/images/'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])

def predict_img(img_path):

    # dimensions of our images    -----   are these then grayscale (black and white)?
    img_width, img_height = 128, 128

    # Get test image ready
    test_image = None
    try:
        test_image = image.load_img(img_path, target_size=(img_width, img_height))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image.reshape(1, img_width, img_height, 3)    # Ambiguity!
    # Should this instead be: test_image.reshape(img_width, img_height, 3) ??
    except:
        return "Get Test Image Error", 200
    result = [2] * 2
    try:
        model = load_model('/home/JingyuXu/result/keras_model.h5')
        proba = model.predict_proba(test_image, batch_size=1)[0][0]
        label = model.predict_classes(test_image, batch_size=1)[0][0]
        if label == 0:
            result[0] = 'benign'
            result[1] = 1 - proba
        elif label == 1:
            result[0] = 'malignant'
            result[1] = proba
        else:
            result[0] = 'unknown'
            result[1] = 0.0
    except:
        result[0], result[1] = 'Load model error', 300 
    finally:
    	keras.backend.clear_session()
    
    print(result)
    return result

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)

app.send_file_max_age_default = timedelta(seconds=1)


# @app.route('/upload', methods=['POST', 'GET'])
@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "Format should be in [png PNG jpg JPG bmp]"})

        user_input = request.form.get("name")

        basepath = os.path.dirname(__file__)

        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))

        f.save(upload_path)

        #Rename file as test.jpg
        #print(secure_filename(f.filename))
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)
        
        #exception
        img_path_new = img_path+secure_filename(f.filename)
            
        label_new = predict_img(img_path_new)
        
        
        return render_template('upload_ok.html', userinput=user_input,
                                                 val1=time.time(), 
                                                 val2=label_new)

    return render_template('upload.html')


if __name__ == '__main__':
    # app.debug = True
    app.run(host="0.0.0.0",port=8080)

