# data.py
import torch

class Dataset:
  def __init__(self, N, min, max):
    self.N = N
    self.min = min
    self.max = max
  
  def generate_data(self):
    # xは-1から1までの一様分布
    x = torch.rand(self.N) * (self.max - self.min) + self.min
    # dは正規分布
    d = torch.randn(self.N) * 0.1
    # yはx + d
    y = x + d
    return x, y
