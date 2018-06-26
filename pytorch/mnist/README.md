# MNIST on pytorch

## 概要

pytorchでMNISTの手書き文字分類を行うニューラルネットワークを実装．

## 使い方

```bash:
python main.py [-h] [--batch-size BATCH_SIZE]
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

モデルの記述．
`def __init__(self)`の中でレイヤの構成．kerasの`Sequential()`のようにボトムからレイヤを追加することができる．畳み込み層など学習パラメタを持つレイヤを記述するのが望ましい．らしい．
`def forward(self, x)`内では，プーリングや活性化関数など単純計算のみで学習パラメタを持たないレイヤを記述する．

```python:
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.linear1 = nn.Linear(64, 1024)
        self.classification = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.dropout(x)
        x = self.classification(x)

        return x
```

### `main.py`

データセット読み込みからのtrainとvalidation．~~kerasより書くのめんどい~~