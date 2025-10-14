import os
import torch
import cv2
import numpy as np
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from ultralytics import YOLO
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
import json

# Load models once when the server starts
def load_models():
    """Load pre-trained models"""
    try:
        # Load YOLO model
        yolo_model = YOLO('yolov8n.pt')
        
        # Load text generation model
        text_model_path = os.path.join(settings.BASE_DIR, 'saved_models', 'text_generator')
        text_tokenizer = GPT2Tokenizer.from_pretrained(text_model_path)
        text_model = GPT2LMHeadModel.from_pretrained(text_model_path)
        
        text_tokenizer.pad_token = text_tokenizer.eos_token
        
        return yolo_model, text_model, text_tokenizer
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

# Load models
yolo_model, text_model, text_tokenizer = load_models()

def home(request):
    """Home page with image upload form"""
    return render(request, 'index.html')

def process_image(request):
    """Process uploaded image and generate description"""
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Save uploaded file
            uploaded_file = request.FILES['image']
            fs = FileSystemStorage()
            
            # Create input directory if not exists
            input_dir = os.path.join(settings.MEDIA_ROOT, 'inputs')
            os.makedirs(input_dir, exist_ok=True)
            
            filename = fs.save(os.path.join('inputs', uploaded_file.name), uploaded_file)
            input_image_path = os.path.join(settings.MEDIA_ROOT, filename)
            
            # Process image with YOLO
            detected_objects = detect_objects(input_image_path)
            
            # Generate description
            description = generate_image_description(detected_objects)
            
            # Create output image with bounding boxes
            output_image_path = create_annotated_image(input_image_path, detected_objects)
            
            # Get relative paths for templates
            input_image_url = os.path.join(settings.MEDIA_URL, filename)
            output_filename = f"annotated_{uploaded_file.name}"
            output_image_url = os.path.join(settings.MEDIA_URL, 'outputs', output_filename)
            
            context = {
                'input_image': input_image_url,
                'output_image': output_image_url,
                'detected_objects': detected_objects,
                'description': description,
                'objects_count': len(detected_objects)
            }
            
            return render(request, 'result.html', context)
        
        except Exception as e:
            error_context = {
                'error': f"Error processing image: {str(e)}"
            }
            return render(request, 'index.html', error_context)
    
    return render(request, 'index.html')

def detect_objects(image_path):
    """Detect objects in image using YOLO"""
    try:
        results = yolo_model(image_path)
        detected_objects = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = yolo_model.names[class_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                
                detected_objects.append({
                    'class': class_name,
                    'confidence': round(confidence, 2),
                    'bbox': [int(coord) for coord in bbox]
                })
        
        return detected_objects
    except Exception as e:
        print(f"Error in object detection: {e}")
        return []

def generate_image_description(detected_objects):
    """Generate descriptive text based on detected objects"""
    if not detected_objects:
        return "No objects detected in the image."
    
    try:
        # Create prompt from detected objects
        objects_list = ", ".join([obj['class'] for obj in detected_objects[:5]])
        prompt = f"In this image, I can see {objects_list}. This scene appears to be"
        
        # Generate description
        inputs = text_tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = text_model.generate(
                inputs,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=text_tokenizer.eos_token_id,
                do_sample=True
            )
        
        description = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return description
    except Exception as e:
        print(f"Error generating description: {e}")
        return "AI description is currently unavailable."

def create_annotated_image(input_path, detected_objects):
    """Create image with bounding boxes and labels"""
    try:
        # Read image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError("Could not read the image file")
        
        # Create output directory
        output_dir = os.path.join(settings.MEDIA_ROOT, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output path
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, f"annotated_{filename}")
        
        # Draw bounding boxes
        for obj in detected_objects:
            bbox = obj['bbox']
            class_name = obj['class']
            confidence = obj['confidence']
            
            # Draw rectangle
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            cv2.rectangle(image, 
                         (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]),
                         (0, 255, 0), -1)
            
            cv2.putText(image, label,
                       (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Save annotated image
        cv2.imwrite(output_path, image)
        return output_path
    except Exception as e:
        print(f"Error creating annotated image: {e}")
        return input_path  # Return original path as fallback