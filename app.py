from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import logging
import gdown

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup logging
logging.basicConfig(level=logging.INFO)

# Model path
MODEL_PATH = "./model/end.h5"
MODEL_DIR = os.path.dirname(MODEL_PATH)

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

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def download_model_if_needed():
    """Download the model if it doesn't exist locally."""
    if not os.path.exists(MODEL_PATH):
        logging.info("Downloading model from Google Drive...")
        url = "https://drive.google.com/uc?id=1w60Z7vMYKSqWhJZQefF8ZqXs6Cv3z90p"
        gdown.download(url, MODEL_PATH, quiet=False)
        logging.info("Model downloaded successfully!")
    else:
        logging.info("Model already exists locally.")

def load_model():
    """Load the trained TensorFlow model."""
    try:
        # Ensure the model is downloaded
        download_model_if_needed()
        
        # Load the model
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logging.info("✅ Model loaded successfully!")
        return model
    except Exception as e:
        logging.error(f"❌ Error loading model: {e}")
        return None

# Load model at startup
model = load_model()

def preprocess_image(image_path):
    """Preprocess the uploaded image."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224), Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        logging.error(f"❌ Error preprocessing image: {e}")
        return None

@app.route("/", methods=["GET"])
def index():
    """Root endpoint to confirm API is running."""
    return jsonify({
        "status": "online",
        "message": "ML Model API is running. POST an image to /predict to classify it."
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Receive an image and return a prediction."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Upload PNG, JPG, or JPEG."}), 400

    # Save the file temporarily
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Preprocess image
    img_array = preprocess_image(filepath)
    if img_array is None:
        return jsonify({"error": "Error processing image"}), 400

    # Predict using the model
    if model is not None:
        predictions = model.predict(img_array)[0]
        class_index = np.argmax(predictions)
        predicted_class = CLASS_LABELS[class_index]
        confidence = float(predictions[class_index])
        
        # Optional: Return all class probabilities
        all_probabilities = {CLASS_LABELS[i]: float(predictions[i]) for i in range(len(CLASS_LABELS))}
    else:
        return jsonify({"error": "Model not loaded properly"}), 500

    # Clean up (remove uploaded image)
    os.remove(filepath)

    return jsonify({
        "result": predicted_class, 
        "confidence": confidence,
        "all_probabilities": all_probabilities
    })

# This is important for Render
if __name__ == "__main__":
    # Use the PORT environment variable provided by Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)