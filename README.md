# Inference Time Comparison

This work has been implemented on an Nvidia Orin device utilizing the Ubuntu 22.04 operating system. 
The project covers a comprehensive range of topics, from the training and registration of models with PyTorch to the conversion of these models to ONNX format and the subsequent testing with ONNX Runtime. 
Furthermore, chapters outlining the conversion of ONNX models to TensorRT and speed benchmarks offer insights into the optimization of performance. In this study, the fastest inference time was measured using the ONNX Runtime, PyTorch, and TensorRT frameworks on the ResNet50 model. 
The results of this study are presented in the accompanying table. It should be noted that this study does not include the display of the classification results of the images; it only measures the inference time.

## Project Structure

- **onnxruntime-cpu.py**: Image classification via ONNX Runtime with CPU Provider.
- **onnxruntime-gpu.py**: Image classification via ONNX Runtime with CUDA Provider.
- **onnxruntime-tensorrt.py**: Image classification via ONNX Runtime with TensorRT Provider.
- **pytroch_classifier.py**: Image classification via PyTorch with CUDA and CPU.
- **pytroch_tensorrt_classifier.py**: Image classification via PyTorch with Torch2trt library.
- **tensorrt_classifier.py**: Image classification via TensorRT.

## Test Configuration (Nvidia Orin Ubuntu 22.04)

| Frameworks                | Version     |
|------------------------|-----------|
| CUDA                   | 12.2     |
| cuDNN                  | 8.9.x    |
| TensorRT               | 8.6.2    |
| PyTorch                | 2.3      |
| Torchvision            | 0.18     |
| Torch2TRT              | 0.5      |
| ONNX                   | 1.16.2   |
| ONNX Runtime-GPU      | 1.17.0   |
| PyCUDA                 | 2024.1.2 | 
| NumPy                  | 1.26.3   |

## Avg Inference Time

| Device                | PyTorch (model_weights.pth) | ONNX Runtime (onnx_model.onnx) | TensorRT (trt_model.trt)      |
|----------------------|------------------------------|----------------------------------|-------------------------------|
| CPU                  | 173.42 ms                   | 40.01 ms                        | -                             |
| CUDA (GPU)          | 13.13 ms                    | 13.98 ms                        | -                             |
| TensorRT (GPU)      | 2.099 ms (torch2trt)        | 3.05 ms (TensorRT Provider)     | 1.98 ms                       |
