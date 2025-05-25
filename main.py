import serial
import requests
import time
import threading
import subprocess # Not used in the provided code, consider removing if truly not needed
import json
import http.server # Not used directly in the client code, consider removing if truly not needed
import socketserver # Not used directly in the client code, consider removing if truly not needed
import cv2
import io
import base64
import pandas as pd # Not used in the provided code, consider removing if truly not needed
from PIL import Image # Added for image processing with PyTorch

import torch
import torch.nn as nn
from torchvision import models, transforms

# --- Configuration ---
# Setup serial port
serial_port = '/dev/ttyAMA0' # Corrected string literal
baud_rate = 115200
try:
    ser = serial.Serial(serial_port, baud_rate, timeout=1)
    print(f"Successfully opened serial port: {serial_port}")
except serial.SerialException as e:
    print(f"Error opening serial port {serial_port}: {e}")
    ser = None # Set ser to None if opening fails

FLASK_SERVER_URL = "http://192.168.22.101:5000"
MODEL_PATH = "pretrained-efficientnet_model.pth"
NUM_CLASSES = 9 # Define the number of classes for your model

# Class names for disease classification
CLASS_NAMES = [
    "bacterial", "downy", "fungal",
    "healthy", "powdery", "septoria",
    "unhealthy", "viral", "wilt"
]

# --- Functions for ESP32 Sensor Data ---
def get_data():
    """Function to read data from serial port of ESP32."""
    if ser is None:
        print("Serial port not initialized. Skipping sensor data reading.")
        return None
    try:
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8').strip()
            print(f"Raw data: {repr(data)}")
            if data:
                # Corrected: data.splot to data.split
                values = data.split(',')
                print(f"Received: {len(values)} values: {values}")
                if len(values) == 6:
                    try:
                        co, co2, lux, dust, temp, wtlv = map(float, values)
                        return {'co': co, 'co2': co2, 'lux': lux, 'dust': dust, 'temp': temp, 'wtlv': wtlv}
                    except ValueError as ve:
                        print(f"Error converting values to float: {ve}. Received values: {values}")
                        return None
                else:
                    print(f"Error: Invalid data format received! Expected 6 values, got {len(values)}. Values: {values}")
                    return None
            else:
                print("Received empty data")
                return None
        else:
            # print("No data available from serial port") # Uncomment for debugging if needed
            return None
    except Exception as e:
        print(f"Error reading from serial: {e}")
        return None

def send_data(data):
    """Function to send sensor data to Flask server."""
    url = f"{FLASK_SERVER_URL}/data"
    try:
        # Corrected: request to requests
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("Sensor data sent successfully")
        else:
            print(f"Failed to send sensor data: {response.status_code}, response: {response.text}")
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error sending sensor data to {url}: {e}")
    except Exception as e:
        print(f"Error sending sensor data: {e}")

def send_sensor_data():
    """Thread target to continuously read and send sensor data."""
    while True:
        data = get_data()
        if data:
            send_data(data) # Corrected: send_data(data): to send_data(data)
        time.sleep(5) # Send sensor data every 5 seconds

# --- AI Model Setup ---
def load_model(model_path, num_classes):
    """Loads the pre-trained EfficientNet model."""
    # Check for CUDA (GPU) and fall back to CPU if not available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.efficientnet_b0(weights=None) # No pretrained weights initially
    # Adjust the classifier layer to match your number of classes
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    
    try:
        # Use map_location to ensure the model loads correctly on CPU if GPU is not available
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval() # Set model to evaluation mode
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please check the path.")
        return None, None
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None
    
    return model, device

# Load the model globally once
model, device = load_model(MODEL_PATH, NUM_CLASSES)
if model is None:
    print("Failed to load AI model. Exiting.")
    exit() # Exit if model loading fails

# Define image transformations for the model
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Functions for Webcam and AI Classification ---
def send_webcam_image_and_prediction():
    """Function to capture image, classify, and send to Flask server."""
    # Initialize webcam
    cap = cv2.VideoCapture(0) # 0 is typically the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read() # Read a frame from the webcam
        if not ret:
            print("Failed to grab frame from webcam. Retrying...")
            time.sleep(1) # Wait a bit before retrying
            continue

        # --- Image Preprocessing for AI Model ---
        # Convert OpenCV BGR image to PIL RGB image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Apply transformations
        input_tensor = test_transform(pil_image).unsqueeze(0).to(device) # Add batch dimension and move to device

        # --- AI Model Inference ---
        predicted_label = "No Prediction"
        try:
            with torch.no_grad(): # Disable gradient calculation for inference
                outputs = model(input_tensor)
                _, pred = torch.max(outputs, 1) # Get the index of the max probability
                predicted_label = CLASS_NAMES[pred.item()] # Get the corresponding label
                print(f"Predicted: {predicted_label}")
        except Exception as e:
            print(f"Error during AI prediction: {e}")
            # Continue even if prediction fails, send image if possible
            
        # --- Prepare Image for Sending (Base64 Encoding) ---
        is_success, buf = cv2.imencode('.jpg', frame)
        if not is_success:
            print("Failed to encode frame as JPEG.")
            time.sleep(1)
            continue
        
        jpg_as_text = base64.b64encode(buf.tobytes()).decode('utf-8') # Convert bytes to string

        # --- Send data to Flask server ---
        url = f"{FLASK_SERVER_URL}/upload_image_and_prediction" # New endpoint for combined data
        payload = {
            'image': jpg_as_text,
            'prediction': predicted_label
        }
        try:
            response = requests.post(url, json=payload, timeout=30) # Add a timeout for requests
            print(f"Server responded: {response.status_code}, {response.text}")
        except requests.exceptions.Timeout:
            print("Request timed out while sending image and prediction.")
        except requests.exceptions.ConnectionError as e:
            print(f"Connection Error sending image and prediction to {url}: {e}")
        except Exception as e:
            print(f"Error sending image and prediction: {e}")
        
        time.sleep(5) # Send image and prediction every 5 seconds (adjust as needed)

    cap.release() # Release the camera when the loop ends (though this loop runs indefinitely)

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Raspberry Pi client...")

    # Create and start threads
    t1 = threading.Thread(target=send_webcam_image_and_prediction, name='send_image_and_prediction')
    t2 = threading.Thread(target=send_sensor_data, name='send_sensor_data')
    
    t1.start()
    t2.start()

    # Optional: Keep the main thread alive, or join threads if they are meant to finish
    try:
        t1.join()
        t2.join()
    except KeyboardInterrupt:
        print("Program terminated by user.")
    finally:
        if ser:
            ser.close()
            print("Serial port closed.")
        print("Exiting.")



