# Consider a dataset for dog and cat detection.
# We aim to build a model for dog and cat detection using transfer learning.

import cv2
import tensorflow as tf
import numpy as np
from preprocess import *
import streamlit as st
# import emoji
#from bs4 import BeautifulSoup
#from urllib.request import urlopen


# def preprocess_images(images):
#     preprocessed_images = []
#     for i in range(len(images)):
#         image = images[i]
        
#         # Convert the BGR image to LAB color space
#         img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         # Apply median filter to the LAB image
#         img = cv2.medianBlur(img, 5)  # You can adjust the kernel size (here, 5) as needed
        
#         # Split the LAB image into L, A, and B channels
#         r, g, b = cv2.split(img)
    
#         # Ensure the channels have the correct data type (8-bit unsigned)
#         r = r.astype(np.uint8)
#         g = g.astype(np.uint8)
#         b = b.astype(np.uint8)
        
#         # Apply CLAHE to each channel separately
#         clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8,8))
#         r = clahe.apply(r)
#         g = clahe.apply(g)
#         b = clahe.apply(b)

#         # Merge the enhanced RGB channels 
#         img_output = cv2.merge([r, g, b])
       
#         # Convert the LAB image back to BGR color space
#         preprocessed_image = cv2.cvtColor(img_output, cv2.COLOR_RGB2BGR)
        
#         preprocessed_images.append(image)
    
#     return np.array(preprocessed_images)

# def cascaded_preprocessing(image):
    
#     # Convert the BGR image to LAB color space
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Apply additional preprocessing to the output of preprocess_images
#     preprocessed_image = preprocess_images(img)
    
#     return preprocessed_image


@st.cache_data
def process_image(file_buffer):
    # Convert the file buffer to a NumPy array
    image = np.frombuffer(file_buffer.getvalue(), dtype=np.uint8)
    # Decode the image
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)  # Use 0 for grayscale
    # Your image processing logic using OpenCV
    # For example, convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
# imshow
def imshow(cv_image, caption="Uploaded Image", use_column_width='auto'):
    st.image(cv_image, caption=caption, use_column_width=use_column_width)
# predict
def predict_image(cv_image):
    resized_image = cv2.resize(cv_image, (224, 224))
    preprocessed_image = cascaded_preprocessing(resized_image)
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
    predicted_class = 'dog' if prediction[0, 0] > 0.5 else 'cat'
    st.info(f'Predicted Class: {predicted_class}')


# Constants
img_size = (224, 224)
img_shape = (img_size[0], img_size[1], 3)

# Load the model
# model = tf.keras.models.load_model("ResNet.h5")
model = tf.keras.models.load_model('ResNet.h5')

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
        predict_image(cv_image)

if __name__ == "__main__":
    main()


# # python -m streamlit run app.py --server.port 8080
