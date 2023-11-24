from PIL import Image
import tensorflow as tf
import numpy as np
import streamlit as st

def preprocess_images(images):
    preprocessed_images = []
    for i in range(len(images)):
        image = images[i]
        
        # Convert the RGB image to LAB color space
        img = Image.fromarray(image, 'RGB')
        
        # Apply median filter to the LAB image
        img = img.filter(ImageFilter.MedianFilter(size=5))
        
        # Split the LAB image into L, A, and B channels
        r, g, b = img.split()
    
        # Apply CLAHE to each channel separately
        clahe = ImageEnhance.Contrast(r).enhance(6.0)
        r = clahe
        clahe = ImageEnhance.Contrast(g).enhance(6.0)
        g = clahe
        clahe = ImageEnhance.Contrast(b).enhance(6.0)
        b = clahe

        # Merge the enhanced RGB channels 
        img_output = Image.merge('RGB', (r, g, b))
       
        # Convert the LAB image back to RGB color space
        preprocessed_image = np.array(img_output)
        
        preprocessed_images.append(preprocessed_image)
    
    return np.array(preprocessed_images)

def cascaded_preprocessing(image):
    # Convert the RGB image to LAB color space
    img = Image.fromarray(image, 'RGB')
    
    # Apply additional preprocessing to the output of preprocess_images
    preprocessed_image = preprocess_images([img])
    
    return preprocessed_image[0]

@st.cache_data
def process_image(file_buffer):
    # Convert the file buffer to a NumPy array
    image = np.array(Image.open(file_buffer))
    # Your image processing logic using Pillow
    # For example, convert to RGB
    image = Image.fromarray(image, 'RGB')
    return np.array(image)
    
# imshow
def imshow(cv_image, caption="Uploaded Image", use_column_width='auto'):
    st.image(cv_image, caption=caption, use_column_width=use_column_width)

# predict
def predict_image(cv_image):
    resized_image = Image.fromarray(cv_image).resize((224, 224))
    preprocessed_image = cascaded_preprocessing(np.array(resized_image))
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
