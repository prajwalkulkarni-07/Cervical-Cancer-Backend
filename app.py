from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model paths
MODEL_PATH = "./model/end.h5"

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
UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the TensorFlow Keras model."""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please upload the model before deployment.")

        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info("✅ Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False

def preprocess_image(image_path):
    """Preprocess the uploaded image."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224), Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        logger.error(f"❌ Error preprocessing image: {e}")
        return None

def predict_with_model(img_array):
    """Make predictions using the Keras model."""
    try:
        predictions = model.predict(img_array)[0]  # Get first batch's prediction
        return predictions
    except Exception as e:
        logger.error(f"❌ Error during prediction: {e}")
        return None

# @app.route("/", methods=["GET"])
# def index():
#     """Root endpoint to confirm API is running."""
#     return jsonify({
#         "status": "online",
#         "message": "ML Model API is running. POST an image to /predict to classify it."
#     })

# @app.route("/", methods=["GET"])
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
    predictions = predict_with_model(img_array)
    
    if predictions is not None:
        class_index = np.argmax(predictions)
        predicted_class = CLASS_LABELS[class_index]
        confidence = float(predictions[class_index])
        
        # Return all class probabilities
        all_probabilities = {CLASS_LABELS[i]: float(predictions[i]) for i in range(len(CLASS_LABELS))}
    else:
        return jsonify({"error": "Model not loaded properly or prediction failed"}), 500

    # Clean up (remove uploaded image)
    os.remove(filepath)

    return jsonify({
        "result": predicted_class, 
        "confidence": confidence,
        "all_probabilities": all_probabilities
    })

# Initialize the model on startup
logger.info("Starting to load model...")
model_loaded = load_model()
logger.info(f"Model loading complete! Success: {model_loaded}")

# Get port from environment variable for Render
port = int(os.environ.get("PORT", 5000))
logger.info(f"Starting app on port {port}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Ensure it uses Render's PORT
    app.run(host="0.0.0.0", port=port)