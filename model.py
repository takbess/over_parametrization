# model.py
import torch

class PolynomialModel(torch.nn.Module):
  def __init__(self, n):
    super().__init__()
    # 重みパラメータ
    self.n = n
    self.w = torch.nn.Parameter(torch.randn(n+1))
  
  def forward(self, x):
    # y = w_0 + w_1 * x + w_2 * x^2 + ... + w_n * x^n
    y = 0
    for i in range(self.n+1):
      y += self.w[i] * x**i
    return y
