from flask import Flask, request, jsonify, render_template_string
import numpy as np
from PIL import Image
import io
import base64
import os
from werkzeug.utils import secure_filename

# Import transformers and related libraries
from transformers import MobileViTImageProcessor, MobileViTForImageClassification
import torch

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables to store the loaded model and processor
model = None
processor = None
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def load_model():
    """Load your trained MobileViT model and processor"""
    global model, processor, class_names
    
    try:
        # Your specific model checkpoint path
        checkpoint_path = r'C:\Users\DELL\Desktop\C++\transformers\checkpoints\mobilevit_xs_best.pt'
        
        # Load the base MobileViT model architecture
        # Using mobilevit-x-small since you mentioned mobilevit_xs
        model = MobileViTForImageClassification.from_pretrained(
            'apple/mobilevit-x-small',
            num_labels=6,  # Your 6 classes
            ignore_mismatched_sizes=True
        )
        
        # Load your fine-tuned weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # If the checkpoint is just the state dict
            model.load_state_dict(checkpoint)
        
        # Load the processor (this handles 224x224 resizing automatically)
        processor = MobileViTImageProcessor.from_pretrained('apple/mobilevit-x-small')
        
        # Set model to evaluation mode
        model.eval()
        
        print("MobileViT model loaded successfully!")
        print(f"Model path: {checkpoint_path}")
        print(f"Input size: 224x224 pixels")
        print(f"Classes: {class_names}")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Please check if the model file exists and is accessible")
        model = None
        processor = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Preprocess image using MobileViT processor"""
    try:
        if processor is None:
            print("Processor not loaded")
            return None
        
        # Convert PIL image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Use the MobileViT processor to preprocess the image
        inputs = processor(images=image, return_tensors="pt")
        
        return inputs
        
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def predict_image(inputs):
    """Make prediction using the MobileViT model"""
    if model is None or processor is None:
        return {"error": "Model or processor not loaded"}
    
    try:
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get predicted class
            predicted_class_idx = torch.argmax(logits, dim=1).item()
            
            # Get confidence (probability) using softmax
            probabilities = torch.softmax(logits, dim=1)
            confidence = float(probabilities[0][predicted_class_idx])
            
            # Get all class probabilities for additional info
            all_probabilities = {
                class_names[i]: float(probabilities[0][i]) 
                for i in range(len(class_names))
            }
            
            predicted_class = class_names[predicted_class_idx]
            
            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_index": predicted_class_idx,
                "all_probabilities": all_probabilities
            }
        
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

def predict_image_with_top_k(inputs, k=3):
    """Make prediction with top-k results"""
    if model is None or processor is None:
        return {"error": "Model or processor not loaded"}
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get probabilities
            probabilities = torch.softmax(logits, dim=1)
            
            # Get top-k predictions
            top_k_probs, top_k_indices = torch.topk(probabilities[0], k)
            
            predictions = []
            for i in range(k):
                predictions.append({
                    "class": class_names[top_k_indices[i].item()],
                    "confidence": float(top_k_probs[i]),
                    "class_index": top_k_indices[i].item()
                })
            
            return {
                "top_predictions": predictions,
                "predicted_class": predictions[0]["class"],
                "confidence": predictions[0]["confidence"]
            }
        
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

# HTML template for file upload interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Image Classification</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
        .upload-area:hover { border-color: #999; }
        .result { margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }
        .error { color: #dc3545; }
        .success { color: #28a745; }
        img { max-width: 300px; max-height: 300px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>Image Classification Service</h1>
    
    <form method="POST" enctype="multipart/form-data" action="/predict">
        <div class="upload-area">
            <p>Select an image file to classify</p>
            <input type="file" name="file" accept="image/*" required>
            <br><br>
            <input type="submit" value="Classify Image" style="padding: 10px 20px; font-size: 16px;">
        </div>
    </form>
    
    <div id="api-info">
        <h2>API Usage</h2>
        <p><strong>Endpoint:</strong> POST /api/predict</p>
        <p><strong>Content-Type:</strong> multipart/form-data</p>
        <p><strong>Parameter:</strong> file (image file)</p>
        <p><strong>Response:</strong> JSON with prediction results</p>
    </div>
</body>
</html>
'''

@app.route('/')
def home():
    """Serve the main page with upload interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict_file():
    """Handle file upload and prediction for web interface"""
    if 'file' not in request.files:
        return render_template_string(HTML_TEMPLATE + '<div class="result error">No file selected</div>')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template_string(HTML_TEMPLATE + '<div class="result error">No file selected</div>')
    
    if file and allowed_file(file.filename):
        try:
            # Read and process image
            image = Image.open(file.stream)
            processed_inputs = preprocess_image(image)
            
            if processed_inputs is None:
                return render_template_string(HTML_TEMPLATE + '<div class="result error">Error processing image</div>')
            
            # Make prediction
            result = predict_image(processed_inputs)
            
            if 'error' in result:
                return render_template_string(HTML_TEMPLATE + f'<div class="result error">{result["error"]}</div>')
            
            # Convert image to base64 for display
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            
            result_html = f'''
            <div class="result success">
                <h3>Prediction Result:</h3>
                <img src="data:image/png;base64,{img_str}" alt="Uploaded image">
                <p><strong>Predicted Class:</strong> {result["predicted_class"]}</p>
                <p><strong>Confidence:</strong> {result["confidence"]:.2%}</p>
            </div>
            '''
            
            return render_template_string(HTML_TEMPLATE + result_html)
            
        except Exception as e:
            return render_template_string(HTML_TEMPLATE + f'<div class="result error">Error: {str(e)}</div>')
    
    return render_template_string(HTML_TEMPLATE + '<div class="result error">Invalid file type</div>')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Process image
        image = Image.open(file.stream)
        processed_inputs = preprocess_image(image)
        
        if processed_inputs is None:
            return jsonify({'error': 'Error processing image'}), 400
        
        # Make prediction
        result = predict_image(processed_inputs)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    processor_status = "loaded" if processor is not None else "not loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'processor_status': processor_status,
        'classes': class_names
    })

@app.route('/api/predict_top_k/<int:k>', methods=['POST'])
def api_predict_top_k(k):
    """API endpoint for top-k predictions"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Process image
        image = Image.open(file.stream)
        processed_inputs = preprocess_image(image)
        
        if processed_inputs is None:
            return jsonify({'error': 'Error processing image'}), 400
        
        # Make top-k prediction
        result = predict_image_with_top_k(processed_inputs, min(k, len(class_names)))
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load the model when starting the app
    load_model()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)