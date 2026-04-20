from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model.h5")

# Class labels (change based on your dataset)
classes = ["Healthy", "Disease1", "Disease2"]

def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def home():
    return "Crop Disease Detection API Running"

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file)
    img = preprocess(img)

    pred = model.predict(img)
    result = classes[np.argmax(pred)]

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)