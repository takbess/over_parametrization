#y=w_0 + w_1 * x + w_2 * x^2 + ... + w_n + x^n
# というモデルが与えられたとき、
# ランダムな(x_n,y_n) のデータ10個を用意して、
# L2 lossと重みを小さくするようなlossを用いて、
# 10 epoch 学習するコードをpytorchで書いて。

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime
import os

# モデルの次数
n = 100

# ランダムデータ10個を生成
x = np.random.uniform(-1,1,10)
y = np.random.uniform(-1,1,10)

# モデル定義
model = nn.Linear(n+1,1)

x = torch.from_numpy(x).float().unsqueeze(1)
y = torch.from_numpy(y).float().unsqueeze(1)

# 特徴量を作成
X = torch.cat([x**i for i in range(n+1)],dim=1)

# 損失関数を定義
mse_loss = nn.MSELoss()

# w_n に関するL1 Loss
alpha = 0.01 # l1_loss にかける係数
l1_loss = nn.L1Loss()

# 最適化手法を定義
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 学習
for epoch in range(100):

    # 勾配をゼロに初期化
    optimizer.zero_grad()

    # モデルの出力を計算
    output = model(X)

    # 損失を計算
    alpha = 0.0 # weightに対するl1_lossの制御パラメタ
    weights = model.parameters()
    loss = 0
    for w in weights:
        loss += alpha * l1_loss(w,torch.zeros_like(w))
    loss += mse_loss(output,y)
    # loss = criterion(output,y)

    # 逆伝播を実行
    loss.backward()

    # パラメータを更新
    optimizer.step()

    # 損失を表示
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 描画

# 描画用のデータ作成
def make_data():
    x = np.linspace(-1.05,1.05,10)
    x = torch.from_numpy(x).float().unsqueeze(1)
    X = torch.cat([x**i for i in range(n+1)],dim=1)

    return x,X
x_plot,X_plot = make_data()
y_plot = model(X_plot)

plt.plot(x_plot,y_plot.detach().numpy(), color="black", label="Model")
plt.scatter(x,y,color="red",label="train data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# 結果の保存
# results/date_*.png と tmp.pntの両方に保存
save_dir = "results/"
os.makedirs(save_dir,exist_ok=True)

now = datetime.datetime.now()
date = now.strftime("%Y%m%d_%H%M%S")
file_name = "date_" + date + ".png"

save_path = os.path.join(save_dir,file_name)
plt.savefig(save_path)
plt.savefig("tmp.png")