import os
import torch
import torchvision
from torchvision import transforms
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model information
MODELS = {
    'breast_cancer': {
        'file': 'models/breast_cancer_model.pth',
        'classes': ['Healthy', 'Sick'],
        'display_name': 'Breast Cancer Detection',
        'description': 'Detects potential breast cancer from histopathology images'
    },
    'covid19': {
        'file': 'models/covid19_model.pth',
        'classes': ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia'],
        'display_name': 'COVID-19 Analysis',
        'description': 'Analyzes chest X-rays for COVID-19 and other lung conditions'
    },
    'malaria': {
        'file': 'models/malaria_model.pth',
        'classes': ['Parasitized', 'Uninfected'],
        'display_name': 'Malaria Detection',
        'description': 'Identifies malaria parasites in blood smear images'
    },
    'pneumonia': {
        'file': 'models/pneumonia_model.pth',
        'classes': ['NORMAL', 'PNEUMONIA'],
        'display_name': 'Pneumonia Detection',
        'description': 'Detects pneumonia in chest X-ray images'
    },
    'skin_cancer': {
        'file': 'models/skin_cancer_model.pth',
        'classes': ['benign', 'malignant'],
        'display_name': 'Skin Cancer Classification',
        'description': 'Classifies skin lesions as benign or malignant'
    },
    'tuberculosis': {
        'file': 'models/tuberculosis_model.pth',
        'classes': ['Normal', 'Tuberculosis'],
        'display_name': 'Tuberculosis Screening',
        'description': 'Screens chest X-rays for signs of tuberculosis'
    }
}

# Initialize model cache
model_cache = {}

def load_model(model_key):
    """Load model if not already in cache"""
    if model_key not in model_cache:
        print(f"Loading model: {model_key}")
        
        # Initialize model architecture (using EfficientNet B0 for all models)
        model = torchvision.models.efficientnet_b0(weights=None)
        num_classes = len(MODELS[model_key]['classes'])
        
        model.classifier = torch.nn.Sequential(
                                    torch.nn.Dropout(p=0.5),
                                    torch.nn.Linear(model.classifier[1].in_features, num_classes)
                                )


        
        # Load the saved weights
        model_path = MODELS[model_key]['file']
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            model_cache[model_key] = model
            print(f"Model {model_key} loaded successfully")
        except Exception as e:
            print(f"Error loading model {model_key}: {str(e)}")
            return None
    
    return model_cache[model_key]

# Define image transformation for inference
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    # Pass model information to the template
    return render_template('index.html', models=MODELS)

@app.route('/api/models')
def get_models():
    return jsonify(MODELS)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    model_key = request.form.get('model', 'skin_cancer')
    
    if model_key not in MODELS:
        return jsonify({'error': 'Invalid model selection'}), 400
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load the selected model
            model = load_model(model_key)
            if model is None:
                return jsonify({'error': f'Failed to load model: {model_key}'}), 500
            
            # Open and transform the image
            img = Image.open(filepath).convert('RGB')
            img_tensor = test_transform(img).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                prediction = torch.argmax(probabilities).item()
            
            # Get class names for this model    
            class_names = MODELS[model_key]['classes']
            
            # Format the result
            result = {
                'prediction': class_names[prediction],
                'confidence': float(probabilities[prediction]) * 100,
                'model_used': model_key,
                'display_name': MODELS[model_key]['display_name'],
                'probabilities': {
                    class_names[i]: float(prob) * 100 
                    for i, prob in enumerate(probabilities)
                }
            }
            
            return jsonify(result)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up - optionally remove the uploaded file after processing
            # os.remove(filepath)
            pass
    
    return jsonify({'error': 'Invalid file type'}), 400

# Route to serve uploaded images (for preview purposes)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Create a JSON file with model info for the frontend
    with open('static/model_info.json', 'w') as f:
        json.dump(MODELS, f)
    
    app.run(debug=True)