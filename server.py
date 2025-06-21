from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
model = load_model('ssrnet_3_3_3_64_1.0_1.0.h5')

def preprocess(img):
    img = img.resize((64, 64))
    img = np.array(img).astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img = Image.open(file.stream).convert("RGB")
    input_tensor = preprocess(img)
    prediction = model.predict(input_tensor)[0][0]
    
    return jsonify({'predicted_age': float(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
