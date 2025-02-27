from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import logging
import gdown  # For Google Drive downloads

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model paths and URLs
MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
# Google Drive file ID extracted from the provided link
GDRIVE_FILE_ID = "1cGT5iRPrtspI5enSwrgZKpIwjayxAZ-S"
GDRIVE_MODEL_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

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

# Create necessary directories
UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def download_model_from_gdrive():
    """Download the ONNX model from Google Drive if it doesn't exist locally."""
    try:
        if os.path.exists(MODEL_PATH):
            logger.info("Model already exists locally.")
            return True
        
        logger.info(f"Downloading model from Google Drive...")
        gdown.download(GDRIVE_MODEL_URL, MODEL_PATH, quiet=False)
        
        if os.path.exists(MODEL_PATH):
            logger.info("✅ Model downloaded successfully!")
            return True
        else:
            logger.error("❌ Failed to download model.")
            return False
    except Exception as e:
        logger.error(f"❌ Error downloading model: {e}")
        return False

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the ONNX model, downloading it first if needed."""
    global session
    try:
        # First ensure we have the model file
        if not download_model_from_gdrive():
            raise FileNotFoundError("Failed to download the model from Google Drive.")

        # Create an ONNX Runtime session
        # For better performance on Render, we can use specific execution providers
        # Default to CPU provider if no GPU is available
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(MODEL_PATH, providers=providers)
        logger.info("✅ ONNX model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading ONNX model: {e}")
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
    """Make predictions using the ONNX model."""
    try:
        # Get the input name(s) from the model
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Run prediction
        predictions = session.run([output_name], {input_name: img_array})[0][0]
        return predictions
    except Exception as e:
        logger.error(f"❌ Error during ONNX prediction: {e}")
        return None

@app.route("/", methods=["GET"])
def index():
    """Root endpoint to confirm API is running."""
    return jsonify({
        "status": "online",
        "message": "ONNX ML Model API is running. POST an image to /predict to classify it."
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for Render."""
    return jsonify({"status": "healthy"}), 200

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

# Initialize the model on startup - only if this is the main app and not being imported
if __name__ == "__main__":
    # Initialize the model on startup - download and load
    logger.info("Starting to download and load ONNX model...")
    model_loaded = load_model()
    logger.info(f"ONNX model initialization complete! Success: {model_loaded}")
    
    # Get port from environment variable for Render
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
else:
    # For gunicorn or other WSGI servers, initialize the model here
    logger.info("Initializing application for WSGI server (Render)...")
    load_model()