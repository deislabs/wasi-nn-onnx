"""
Usage: pip install torch

python identity_model.py
"""
import torch 

__version__ = '0.1.0'

class Model(torch.nn.Module):
  def forward(self, x):
    return x

m = Model()
x = torch.randn(1, 4)
torch.onnx.export(m, (x, ), 'identity_input_output.onnx')
