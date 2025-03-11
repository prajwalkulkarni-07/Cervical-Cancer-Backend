from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import logging
from flask_cors import CORS  # Add this import

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup logging
logging.basicConfig(level=logging.INFO)

# Model path
MODEL_PATH = "./end.h5"
UPLOAD_FOLDER = "static/uploads/"
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

@app.route("/predict", methods=["POST"])  # Changed route to /predict
def predict():
    if "image" not in request.files:  # Changed from 'file' to 'image' to match frontend
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Upload PNG, JPG, or JPEG."}), 400
    
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    img_array = preprocess_image(filepath)
    if img_array is None:
        return jsonify({"error": "Error processing image"}), 400

    if model is not None:
        predictions = model.predict(img_array)[0]  # Get class probabilities
        class_index = np.argmax(predictions)  # Get class index
        predicted_class = CLASS_LABELS[class_index]  # Get class label
        confidence = float(predictions[class_index])  # Get confidence as float
        
        # Return JSON response
        return jsonify({
            "result": predicted_class,
            "confidence": confidence
        })
    else:
        return jsonify({"error": "Model not loaded properly"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)