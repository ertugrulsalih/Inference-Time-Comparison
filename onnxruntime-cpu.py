import onnxruntime as ort
import numpy as np
from PIL import Image
import time
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# Load ONNX model
ort_session = ort.InferenceSession('models/onnx_model.onnx')

# Load ResNet50 class labels
weights = ResNet50_Weights.DEFAULT
class_names = weights.meta["categories"]

# Define preprocessing steps
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

# Function for image classification using ONNX (CPU)
def classify_image_onnx_cpu(image_path):

    # Load and preprocess the image
    img = Image.open(image_path)
    img_t = transform(img).unsqueeze(0).numpy()

    # Measure inference time
    start_time = time.time()
    N = 1000  # Run inference multiple times for averaging
    for _ in range(N):
        ort_inputs = {ort_session.get_inputs()[0].name: img_t}
        ort_outs = ort_session.run(None, ort_inputs)
    end_time = time.time()

    # Calculate average inference time
    avg_time = (end_time - start_time) / N
    print(f"Avg Inference Time (CPU Provider): {(avg_time*1000):.6f} ms")

# Classify the input image
classify_image_onnx_cpu("images/Cat_November_2010-1a.jpg")
