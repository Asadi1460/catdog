import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import streamlit as st

# Define the model
class DogCatClassifier(nn.Module):
    def __init__(self):
        super(DogCatClassifier, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.fc = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

# Load the pre-trained model
model = DogCatClassifier()
model.eval()

# Define transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@st.cache
def process_image(file_buffer):
    # Convert the file buffer to a PIL Image
    pil_image = Image.open(file_buffer).convert("RGB")

    # Apply transformations
    tensor_image = preprocess(pil_image)
    tensor_image = torch.unsqueeze(tensor_image, 0)

    return tensor_image

# Display the image
def imshow(pil_image, caption="Uploaded Image", use_column_width='auto'):
    st.image(pil_image, caption=caption, use_column_width=use_column_width)

# Predict the image
def predict_image(tensor_image):
    with torch.no_grad():
        prediction = torch.sigmoid(model(tensor_image))

    predicted_class = 'dog' if prediction.item() > 0.5 else 'cat'
    st.info(f'Predicted Class: {predicted_class}')

def main():
    st.title("Classify CAT & Dog")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Process the image
        tensor_image = process_image(uploaded_file)

        # Display the original image        
        imshow(uploaded_file, caption="Uploaded Image")
        # Predict the image
        predict_image(tensor_image)

if __name__ == "__main__":
    main()
