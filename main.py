# main.py
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import loss as losslib
from omegaconf import DictConfig, OmegaConf
import os

config_path = "conf/config.yaml"
dir = os.path.dirname(config_path)
file_name = os.path.basename(config_path)

@hydra.main(config_path=dir, config_name=file_name, version_base=None)
def main(cfg: DictConfig):
  # mlflowの実験を開始
  mlflow.start_run()
  
  # ハイパラ保存
  for key, value in cfg.items():
    mlflow.log_param(key, value)


  # データセットの作成
  dataset = hydra.utils.instantiate(cfg.dataset)
  x_train, y_train = dataset.generate_data()
  x_test, y_test = dataset.generate_data()
  # モデルの作成
  model = hydra.utils.instantiate(cfg.model)
  # ロス関数の作成
  loss_fn = hydra.utils.instantiate(cfg.loss)
  # オプティマイザの作成
  optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())

  # 学習のループ
  for epoch in range(cfg.trainer.epochs):
    # 勾配をゼロにする
    optimizer.zero_grad()
    # モデルの出力
    y_pred = model(x_train)
    # lossの計算
    loss = loss_fn(y_pred, y_train, model.w)
    # 勾配の計算
    loss.backward()
    # パラメータの更新
    optimizer.step()
    # testデータでの評価
    y_test_pred = model(x_test)
    test_loss = losslib.l2_loss(y_test_pred, y_test)
    # lossの表示
    print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
    # mlflowにメトリクスを記録
    mlflow.log_metric("train_loss", loss.item(), step=epoch)
    mlflow.log_metric("test_loss", test_loss.item(), step=epoch)

    if epoch % 100 == 0:
      # モデルのプロット
      # プロット用のx
      # x_plot = torch.linspace(x_train.min().item(), x_train.max().item(), 100)
      x_plot = torch.linspace(-1, 1, 100)
      # モデルの出力
      y_plot = model(x_plot)
      # グラフの描画
      plt.plot(x_plot.numpy(), y_plot.detach().numpy(), color="black", label="Model")
      plt.scatter(x_train, y_train, color="red", label="Train Data")
      plt.scatter(x_test, y_test, color="blue", label="Test Data")
      plt.xlabel("x")
      plt.ylabel("y")
      plt.legend()
      # plt.show()
      plt.ylim([-2, 2])
      plt.savefig("tmp_change_epochs.png")
      print("save figure in tmp.png")
      plt.clf()
      

  # 最終的なlossの保存
  # with open("all_loss.txt","a") as f:
  #   f.write(f"cfg.model.n: {cfg.model.n}, test_loss: {test_loss.item()} \n")
  with open("all_loss.txt","a") as f:
    f.write(f"{cfg.model.n}, {test_loss.item()} \n")
  
  # mlflowにモデルを保存
  mlflow.pytorch.log_model(model, "model")
  # mlflowの実験を終了
  mlflow.end_run()

  # モデルのプロット
  # プロット用のx
  # x_plot = torch.linspace(x_train.min().item(), x_train.max().item(), 100)
  x_plot = torch.linspace(-1, 1, 100)
  # モデルの出力
  y_plot = model(x_plot)
  # グラフの描画
  plt.plot(x_plot.numpy(), y_plot.detach().numpy(), color="black", label="Model")
  plt.scatter(x_train, y_train, color="red", label="Train Data")
  plt.scatter(x_test, y_test, color="blue", label="Test Data")
  plt.xlabel("x")
  plt.ylabel("y")
  plt.legend()
  # plt.show()
  plt.savefig("tmp.png")
  print("save figure in tmp.png")
  plt.clf()

if __name__ == "__main__":
  main()
