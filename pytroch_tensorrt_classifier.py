import torch
from torch2trt import torch2trt
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import time

# Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained ResNet50 model and move it to the selected device
weights = ResNet50_Weights.DEFAULT
model = torch.load('models/model_weights.pth').to(device)
model.eval()  

# Convert the model to TensorRT for faster inference on GPU
model_trt = torch2trt(model, [torch.randn(1, 3, 224, 224).to(device)])


img = Image.open("images/Cat_November_2010-1a.jpg")
preprocess = weights.transforms() 
batch = preprocess(img).unsqueeze(0).to(device) 

# 1. Measure inference time using the regular Torch CUDA model
start_time = time.time()
N = 1000  # Run the inference N times to calculate the average time
for _ in range(N):
    with torch.no_grad():  
        outputs = model(batch)
end_time = time.time()

# Calculate the average inference time for Torch
avg_time_torch = (end_time - start_time) / N
print(f"Torch CUDA Inference time: {(1000*avg_time_torch):.6f} ms")

# 2. Measure inference time using the TensorRT-optimized model
start_time = time.time()
for _ in range(N):
    with torch.no_grad():
        outputs_trt = model_trt(batch)
end_time = time.time()

# Calculate the average inference time for TensorRT
avg_time_trt = (end_time - start_time) / N
print(f"Torch2TRT Inference time: {(1000*avg_time_trt):.6f} ms")

# Compare the two methods and print the speedup factor
speedup = avg_time_torch / avg_time_trt
print(f"Inference Speedup: {speedup:.2f}x")
