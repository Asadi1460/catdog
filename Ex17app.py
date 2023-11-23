# Consider a dataset for dog and cat detection.
# We aim to build a model for dog and cat detection using transfer learning.

import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from preprocess import *

# Constants
img_size = (224, 224)
img_shape = (img_size[0], img_size[1], 3)


# Load the model
model = tf.keras.models.load_model("ResNet.h5")

image_path = 'cat.jpeg'

input_image = cv2.imread(image_path)
resized_image = cv2.resize(input_image, (224, 224))

preprocessed_image = cascaded_preprocessing(resized_image)

prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))

predicted_class = 'dog' if prediction[0, 0] > 0.5 else 'cat'

recolor_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

plt.imshow(recolor_image)
plt.title(f'Predicted Class: {predicted_class}', color='blue', fontsize=12)
plt.axis('off')
plt.show()
