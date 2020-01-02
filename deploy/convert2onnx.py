# Build a Mock Model in PyTorch with a convolution and a reduceMean layer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.onnx as torch_onnx

from models.get_model import get_model
from utils.opts import Opt


opt = Opt().parse()
opt.model = 'YOLO-Nano-onnx'
opt.batch_size = 1

model = get_model(opt)
print('loading checkpoint {}'.format(opt.resume_path))
checkpoint = torch.load('path/to/your.pth')
model.load_state_dict(checkpoint['state_dict'])

model = model.eval()
model.train(False)
model = model.to(torch.device('cuda'))

# Use this an input trace to serialize the model
input_shape = (3, 416, 416)
dummy_input = Variable(torch.randn(opt.batch_size, *input_shape, device='cuda'))

model_onnx_path = "yolo-nano.onnx"

# Export the model to an ONNX file
output = torch_onnx.export(model,
                          dummy_input,
                          model_onnx_path,
                          opset_version=9,
                          verbose=True,
                          export_params=True)
print("Export of yolo-nano.onnx complete!")
