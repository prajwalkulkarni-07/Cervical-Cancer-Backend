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
logger = logging.getLogger(__name__)

# Model paths
MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "end.h5")
TFLITE_PATH = os.path.join(MODEL_DIR, "end.tflite")

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
        logger.info("Downloading model from Google Drive...")
        url = "https://drive.google.com/uc?id=1w60Z7vMYKSqWhJZQefF8ZqXs6Cv3z90p"
        gdown.download(url, MODEL_PATH, quiet=False)
        logger.info("Model downloaded successfully!")
    else:
        logger.info("Model already exists locally.")

def convert_to_tflite():
    """Convert TensorFlow model to TensorFlow Lite for memory efficiency."""
    if not os.path.exists(TFLITE_PATH):
        logger.info("Converting model to TensorFlow Lite...")
        try:
            # Load the original model
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            
            # Convert to TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            
            # Save the TFLite model
            with open(TFLITE_PATH, 'wb') as f:
                f.write(tflite_model)
                
            logger.info("Model converted successfully!")
            
            # Clear the original model from memory
            del model
            import gc
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error converting model: {e}")

# Global variables for TFLite model
interpreter = None
input_details = None
output_details = None

def load_model():
    """Load the TensorFlow Lite model."""
    global interpreter, input_details, output_details
    
    try:
        # Ensure model is downloaded and converted
        download_model_if_needed()
        convert_to_tflite()
        
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info("✅ TFLite model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading TFLite model: {e}")
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

def predict_with_tflite(img_array):
    """Make predictions using the TFLite model."""
    if interpreter is None:
        return None
    
    try:
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        return predictions
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
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

    # Predict using the TFLite model
    predictions = predict_with_tflite(img_array)
    
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
    app.run(host="0.0.0.0", port=port)