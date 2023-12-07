# loss.py
import torch

# L2 lossの定義
def l2_loss(y_pred, y_true):
  return torch.mean((y_pred - y_true)**2)

# 重みの正則化項の定義
def regularization_loss(w):
  return torch.sum(w**2)

# 全体のlossの定義
# def total_loss(y_pred, y_true, w, alpha):
#   # alphaは正則化項の重み
#   return l2_loss(y_pred, y_true) + alpha * regularization_loss(w)

class total_loss:
  def __init__(self, alpha):
    self.alpha = alpha

  def __call__(self,y_pred,y_true,w):
    return l2_loss(y_pred, y_true) + self.alpha * regularization_loss(w)
