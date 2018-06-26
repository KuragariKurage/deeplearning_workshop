# CIFAR10 on pytorch

## 概要

pytorchでCIFAR10の画像分類タスクを行うCNNの実装．

## 使い方

mnistと同じ．

```bash:
main.py [-h] [--batch-size BATCH_SIZE]
               [--test-batch-size TEST_BATCH_SIZE] [--epochs N] [--lr LR]
               [--momentum MOMENTUM] [--no-cuda] [--seed S] [--log-interval N]
```

optional arguments:
  - `-h, --help`            ヘルプの表示
  - `--batch-size BATCH_SIZE`
                        学習に用いるミニバッチ数
  - `--test-batch-size TEST_BATCH_SIZE`
                        検証に用いるミニバッチ数
  - `--epochs N`            学習エポック数
  - `--lr LR`               学習率
  - `--momentum MOMENTUM`   モメンタム
  - `--no-cuda`             CUDAを使用しない（CPUで学習するとき）
  - `--seed S`              ランダムシード
  - `--log-interval N`      エポックNごとにログを表示


## コード内容

### `network.py`


### `main.py`