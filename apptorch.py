import cv2
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Define the preprocessing transformations using torchvision
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def preprocess_images(images):
    preprocessed_images = []
    for i in range(len(images)):
        image = images[i]
        
        # Convert the BGR image to RGB
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply median filter to the image
        img = cv2.medianBlur(img, 5)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(img)
        
        # Apply CLAHE
        pil_image = pil_image.convert('LAB')
        clahe = transforms.ColorJitter(clip_limit=6.0)
        pil_image = clahe(pil_image)
        
        # Convert back to numpy array
        preprocessed_image = np.array(pil_image)
        
        preprocessed_images.append(preprocessed_image)
    
    return np.array(preprocessed_images)

def cascaded_preprocessing(image):
    preprocessed_image = preprocess_images([image])[0]
    return preprocessed_image

@st.cache
def process_image(file_buffer):
    # Convert the file buffer to a NumPy array
    image = np.frombuffer(file_buffer.getvalue(), dtype=np.uint8)
    # Decode the image
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # Your image processing logic using OpenCV
    # For example, convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def imshow(cv_image, caption="Uploaded Image", use_column_width='auto'):
    st.image(cv_image, caption=caption, use_column_width=use_column_width)

def predict_image(cv_image, model):
    resized_image = cv2.resize(cv_image, (224, 224))
    preprocessed_image = cascaded_preprocessing(resized_image)
    
    # Convert to PyTorch tensor
    input_tensor = preprocess(preprocessed_image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor)
    
    predicted_class = 'dog' if prediction[0, 0] > 0.5 else 'cat'
    st.info(f'Predicted Class: {predicted_class}')

# Load the PyTorch model
model = torch.load('ResNet.h5', map_location=torch.device('cpu'))
model.eval()

def main():
    st.title("Classify CAT & Dog")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Process the image
        cv_image = process_image(uploaded_file)
        
        # Display the original image        
        imshow(cv_image)
        # predict the image
        predict_image(cv_image, model)

if __name__ == "__main__":
    main()
