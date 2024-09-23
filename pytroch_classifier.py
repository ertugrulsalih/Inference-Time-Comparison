import torch
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import time

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet50 model with pre-trained weights
weights = ResNet50_Weights.DEFAULT
model = torch.load('models/model_weights.pth').to(device)
model.eval()  # Set model to evaluation mode
print(f"Model is on device: {next(model.parameters()).device}")

# Load and preprocess the input image
img = Image.open("images/Cat_November_2010-1a.jpg")
preprocess = weights.transforms()  # Use default transforms for ResNet50
batch = preprocess(img).unsqueeze(0).to(device)  # Prepare image batch for the model

# Measure inference time
start_time = time.time()
N = 1000  # Run inference multiple times to calculate the average time
for _ in range(N):
  with torch.no_grad():  # Disable gradient calculation for faster inference
    outputs = model(batch)
end_time = time.time()

# Calculate average inference time
avg_time = (end_time - start_time) / N

# Print the average inference time
print(f"Inference time: {(1000*avg_time):.6f} ms")
