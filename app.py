from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import logging

# Initialize Flask app
app = Flask(__name__)
run_with_ngrok(app)  # This will start ngrok when you run the app

# Setup logging
logging.basicConfig(level=logging.INFO)

# Model path (ensure your model is accessible or upload it to Colab)
MODEL_PATH = "/content/drive/MyDrive/Cervical Cancer Website/Ensemble.h5"
UPLOAD_FOLDER = "/content/drive/MyDrive/Cervical Cancer Website/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Define class labels
CLASS_LABELS = [
    "Carcinoma In Situ",
    "Light Dysplastic",
    "Moderate Dysplastic",
    "Normal Columnar",
    "Normal Intermediate",
    "Normal Superficial",
    "Severe Dysplastic"
]

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the model efficiently
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logging.info("✅ Model loaded successfully!")
        return model
    except Exception as e:
        logging.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224), Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        logging.error(f"❌ Error preprocessing image: {e}")
        return None

@app.route("/", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No image uploaded", 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return "Invalid file type. Upload PNG, JPG, or JPEG.", 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    img_array = preprocess_image(filepath)
    if img_array is None:
        return "Error processing image", 400

    if model is not None:
        predictions = model.predict(img_array)[0]
        class_index = np.argmax(predictions)
        predicted_class = CLASS_LABELS[class_index]
        confidence = float(predictions[class_index])
    else:
        return "Model not loaded properly", 500

    return jsonify(result=predicted_class, confidence=confidence)

if __name__ == "__main__":
    app.run()