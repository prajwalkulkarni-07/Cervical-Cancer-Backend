from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import logging
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup logging
logging.basicConfig(level=logging.INFO)

# Model path (Render clones the repo, so the model will be present)
MODEL_PATH = "./DenseNet121.h5"
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

# Load the model
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logging.info("✅ Model loaded successfully!")
        return model
    except Exception as e:
        logging.error(f"❌ Error loading model: {e}")
        return None

# Load model globally
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

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
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
        predictions = model.predict(img_array)[0]
        class_index = np.argmax(predictions)
        predicted_class = CLASS_LABELS[class_index]
        confidence = float(predictions[class_index])
        
        return jsonify({
            "result": predicted_class,
            "confidence": confidence
        })
    else:
        return jsonify({"error": "Model not loaded properly"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render assigns a port dynamically
    app.run(host="0.0.0.0", port=port)
