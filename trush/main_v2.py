# main.py # 色々とミスってる
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import loss as losslib

def log_omegaconf_mlflow(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)

def _explore_recursive(parent_name,element):
    if isinstance(element,DictConfig):
        for k,v in element.items():
            if isinstance(v,DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}',v)
            else:
                mlflow.log_param(f'{parent_name}.{k}',v)
    elif isinstance(element, ListConfig): # ListConfig まだ出てないから挙動が不明
        for i,v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}',v)
    else:
        mlflow.log_param(f'{parent_name}',element)


@hydra.main(config_path="conf", version_base=None, config_name="config")
def main(cfg: DictConfig):
  
  hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
  mlflow.set_experiment(cfg.mlflow.runname)
  # mlflowの実験を開始
  mlflow.start_run()

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
    log_omegaconf_mlflow(cfg)
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

  # mlflowにモデルを保存
  mlflow.pytorch.log_model(model, "model")
  # mlflowの実験を終了
  mlflow.end_run()

  # モデルのプロット
  # プロット用のx
  x_plot = torch.linspace(x_train.min().item(), x_train.max().item(), 100)
  # モデルの出力
  y_plot = model(x_plot)
  # グラフの描画
  plt.plot(x_plot.numpy(), y_plot.detach().numpy(), color="black", label="Model")
  plt.scatter(x_train, y_train, color="red", label="Train Data")
  plt.scatter(x_test, y_test, color="blue", label="Test Data")
  plt.xlabel("x")
  plt.ylabel("y")
  plt.legend()
  plt.show()
  plt.savefig("tmp.png")

if __name__ == "__main__":
  main()
