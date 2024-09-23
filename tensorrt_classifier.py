import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import time
from torchvision import transforms
from torchvision.models import ResNet50_Weights

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Function to load the TensorRT engine
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Load TensorRT engine and create execution context
engine = load_engine('models/trt_model.trt')
context = engine.create_execution_context()

# Load class labels
weights = ResNet50_Weights.DEFAULT
class_names = weights.meta["categories"]

# Image processing transforms
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

# Function to manage CUDA memory
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        buffer = cuda.mem_alloc(size * dtype().itemsize)
        bindings.append(int(buffer))
        if engine.binding_is_input(binding):
            inputs.append(buffer)
        else:
            outputs.append(buffer)
    return inputs, outputs, bindings, stream

# Function to classify an image using TensorRT
def classify_image_tensorrt(image_path):
    # Load and preprocess the image
    img = Image.open(image_path)
    img_t = transform(img).unsqueeze(0).numpy().astype(np.float32)

    # Set up input/output buffers for TensorRT
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Copy input to GPU memory
    cuda.memcpy_htod_async(inputs[0], img_t, stream)

    # Measure inference time
    start_time = time.time()
    N = 1000  # Loop 1000 times
    for _ in range(N):
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Copy output to CPU
    output_shape = engine.get_binding_shape(1)
    output = np.empty(trt.volume(output_shape), dtype=np.float32)
    cuda.memcpy_dtoh_async(output, outputs[0], stream)
    stream.synchronize()  # Synchronize the asynchronous operation
    end_time = time.time()

    # Calculate average inference time
    avg_time = (end_time - start_time) / N
    print(f"Avg Inference Time (TensorRT): {(avg_time * 1000):.6f} ms")

# Classify an image for testing
classify_image_tensorrt("images/Cat_November_2010-1a.jpg")
