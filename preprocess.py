import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def preprocess_images(images):
    preprocessed_images = []
    for i in range(len(images)):
        image = images[i]
        
        # Convert the BGR image to LAB color space
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply median filter to the LAB image
        img = cv2.medianBlur(img, 5)  # You can adjust the kernel size (here, 5) as needed
        
        # Split the LAB image into L, A, and B channels
        r, g, b = cv2.split(img)
    
        # Ensure the channels have the correct data type (8-bit unsigned)
        r = r.astype(np.uint8)
        g = g.astype(np.uint8)
        b = b.astype(np.uint8)
        
        # Apply CLAHE to each channel separately
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8,8))
        r = clahe.apply(r)
        g = clahe.apply(g)
        b = clahe.apply(b)

        # Merge the enhanced RGB channels 
        img_output = cv2.merge([r, g, b])
       
        # Convert the LAB image back to BGR color space
        preprocessed_image = cv2.cvtColor(img_output, cv2.COLOR_RGB2BGR)
        
        preprocessed_images.append(image)
    
    return np.array(preprocessed_images)

def cascaded_preprocessing(image):
    
    # Convert the BGR image to LAB color space
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply additional preprocessing to the output of preprocess_images
    preprocessed_image = preprocess_images(img)
    
    return preprocessed_image
