from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array

from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
from flask import Flask,jsonify,request
import io


app = Flask(__name__)

model = None

def load_model():
    global model
    model = ResNet50(weights='imagenet')

def prepare_image(image,target):

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image

@app.route("/predict",methods=['POST'])
def predict():

    data = {'success':False}

    if request.method == 'POST':
        if request.files.get('image'):

            image = request.files['image'].read()
            image = Image.open(io.BytesIO(image))

            image = prepare_image(image,target=(224,224))

            preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)

            data['predictions'] = []

            for (imagenetID,label,prob) in results[0]:
                r = {'label':label,'probablity':float(prob)}
                data['predictions'].append(r)

            data['success'] = True

    
    return jsonify(data)

if __name__=='__main__':
    print("Loading model....")
    load_model()
    app.run(debug=False,threaded=False,port=3000)
         