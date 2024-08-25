from PIL import Image
import torch
from torchvision import transforms
import numpy as np

# Define the transformation: resize, crop, and normalize
preprocess = transforms.Compose(
    [
        transforms.Resize(256),  # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),  # Crop the center 224x224 pixels
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(  # Normalize the tensor
            mean=[0.485, 0.456, 0.406],  # Mean values for each channel
            std=[0.229, 0.224, 0.225],  # Standard deviation for each channel
        ),
    ]
)

# Load your image
input_image = Image.open("/home/mohebifarimah/EzPC/Athos/Networks/AlexNet/alexnet.jpeg")

# Apply the transformation
input_tensor = preprocess(input_image)

# Add a batch dimension (needed for the model input)
input_batch = input_tensor.unsqueeze(0)  # Shape: [1, 3, 224, 224]

# Ensure the tensor is in the correct format
input_batch = input_batch.to(torch.float32)

# Convert the tensor to a NumPy array
input_array = input_batch.numpy()

# Save the array as a .npy file
np.save("preprocessed_image.npy", input_array)

print("Image saved as preprocessed_image.npy")
