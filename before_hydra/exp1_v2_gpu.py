# pytorch を用いて以下のようなコードを書いてください。

# 1. データセット作成
# (x_n,y_n) のtrain データ10個、testデータ10個を用意する。
# y= x + d
# に従う(x,y)を作成する。ここでdは正規分布である。

# 2. モデル作成
# y=w_0 + w_1 * x + w_2 * x^2 + ... + w_n + x^n
# というモデルを作成する。

# 3. lossの作成
# L2 loss と重みが大きくなりすぎないようにするlossを合わせたものを作成する。

# 4. 学習
# SGDで 10 epoch 学習するコードを作成する。
# 毎epochごとにtestデータでの評価も実行する。

# 5. モデルのプロット
# 学習後の
# y=w_0 + w_1 * x + w_2 * x^2 + ... + w_n + x^n
# のグラフと、
# (x_n,y_n) のtrain データと、
# test データを
# それぞれ黒、赤、青でプロットする。

import torch
import numpy as np
import matplotlib.pyplot as plt
import time 

start_time = time.time()

# 0. cuda 利用
if torch.cuda.is_available():
  device = torch.device("cuda:0")
else:
  device = torch.device("cpu")

# 1. データセット作成
# データ数
N = 10

# データ生成関数
def generate_data(N,min=-1,max=1):
  # xは-1から1までの一様分布
  x = torch.rand(N) * (max - min) + min
  # dは正規分布
  d = torch.randn(N) * 0.1
  # yはx + d
  y = x + d
  return x, y

# trainデータとtestデータを作成
x_train, y_train = generate_data(N)
x_test, y_test = generate_data(N,min=-1,max=1)
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

# 2. モデル作成
# モデルの次数
n = 1000

# モデルの定義
class PolynomialModel(torch.nn.Module):
  def __init__(self, n):
    super().__init__()
    # 重みパラメータ
    self.w = torch.nn.Parameter(torch.randn(n+1))
  
  def forward(self, x):
    # y = w_0 + w_1 * x + w_2 * x^2 + ... + w_n * x^n
    y = 0
    for i in range(n+1):
      y += self.w[i] * x**i
    return y

# モデルのインスタンス化
model = PolynomialModel(n)
model = model.to(device)

# 3. lossの作成
# L2 lossの定義
def l2_loss(y_pred, y_true):
  return torch.mean((y_pred - y_true)**2)

# 重みの正則化項の定義
def regularization_loss(w):
  return torch.sum(w**2)

# 全体のlossの定義
def total_loss(y_pred, y_true, w, alpha):
  # alphaは正則化項の重み
  return l2_loss(y_pred, y_true) + alpha * regularization_loss(w)

# 4. 学習
# 学習率
lr = 0.01
# 正則化項の重み
alpha = 0.0
# オプティマイザ
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# エポック数
epochs = 1000

# 学習のループ
for epoch in range(epochs):
  # 勾配をゼロにする
  optimizer.zero_grad()
  # モデルの出力
  y_pred = model(x_train)
  # lossの計算
  loss = total_loss(y_pred, y_train, model.w, alpha)
  # 勾配の計算
  loss.backward()
  # パラメータの更新
  optimizer.step()
  # testデータでの評価
  y_test_pred = model(x_test)
  test_loss = l2_loss(y_test_pred, y_test)
  # lossの表示
  print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

end_time = time.time()
total_time = end_time - start_time
print(f"学習にかかったそう時間は {total_time:.2f}秒です。")

# 5. モデルのプロット
# プロット用のx
# x_plot = torch.linspace(-1, 1, 100)
x_plot = torch.linspace(-0.75, 0.75, 100)
x_plot = torch.linspace(x_train.min().item(), x_train.max().item(), 100)

# モデルの出力
y_plot = model(x_plot).cpu()
# グラフの描画
plt.plot(x_plot.cpu().numpy(), y_plot.detach().numpy(), color="black", label="Model")
plt.scatter(x_train.cpu(), y_train.cpu(), color="red", label="Train Data")
plt.scatter(x_test.cpu(), y_test.cpu(), color="blue", label="Test Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
plt.savefig("tmp_v2.png")