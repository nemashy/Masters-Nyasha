# Convert the PyTorch model to ONNX format

import torch
import torch.onnx

# weights_path = "C:/Users/nyasha/Desktop/Masters-Nyasha/Training/model_cnn.pth"
# state_dict = torch.load(weights_path)


#  Initialize model in TensorRT

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt