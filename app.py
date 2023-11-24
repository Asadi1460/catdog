# Consider a dataset for dog and cat detection.
# We aim to build a model for dog and cat detection using transfer learning.

import cv2
import tensorflow as tf
import numpy as np
from preprocess import *
import streamlit as st


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
