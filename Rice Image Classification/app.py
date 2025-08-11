import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Paths to model and model info
MODEL_PATH = 'C:\\Users\\DELL\\Desktop\\C++\\Rice\\best_rice_model.keras'
MODEL_INFO_PATH = 'C:\\Users\\DELL\\Desktop\\C++\\Rice\\model_info.json'
UPLOAD_FOLDER = 'Uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model_and_info(model_path, model_info_path):
    """Load the .keras model and model info (class indices, image size)."""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    try:
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        print(f"Model info loaded successfully from {model_info_path}")
    except Exception as e:
        print(f"Error loading model info: {e}")
        raise

    return model, model_info

def preprocess_image(image_path, target_size):
    """Preprocess a single image for prediction."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Rescale to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        print(f"Preprocessed image {image_path} with shape {img_array.shape}")
        return img_array
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

# Load model and info at startup
model, model_info = load_model_and_info(MODEL_PATH, MODEL_INFO_PATH)

@app.route('/')
def index():
    """Serve the React frontend."""
    static_file = 'index.html'
    static_path = os.path.join(app.static_folder, static_file)
    print(f"Attempting to serve: {static_path}")
    if not os.path.exists(static_path):
        print(f"File not found: {static_path}")
        return jsonify({'error': 'index.html not found in static folder'}), 404
    return app.send_static_file(static_file)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction."""
    print("Received request to /predict")
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        print("No file selected")
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        print(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type. Only JPG, JPEG, PNG allowed'}), 400

    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"Saving file to {file_path}")
    file.save(file_path)

    # Preprocess and predict
    img_height = model_info['img_height']
    img_width = model_info['img_width']
    class_names = model_info['class_names']
    print(f"Preprocessing with target size: ({img_width}, {img_height})")
    img_array = preprocess_image(file_path, (img_width, img_height))
    
    if img_array is None:
        print("Preprocessing failed")
        os.remove(file_path)  # Clean up
        return jsonify({'error': 'Error preprocessing image'}), 500

    try:
        print("Running model prediction")
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
        }
        print(f"Prediction result: {result}")
        os.remove(file_path)  # Clean up
        return jsonify(result), 200
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        os.remove(file_path)  # Clean up
        return jsonify({'error': f'Error making prediction: {str(e)}'}), 500

if __name__ == '__main__':
    print(f"Static folder: {app.static_folder}")
    app.run(debug=True, host='0.0.0.0', port=5000)