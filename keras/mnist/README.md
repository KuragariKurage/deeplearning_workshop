# 使い方

## 概要
MNISTデータセットを用いて、手書き文字を識別するネットワークを作成し、テストも行う。

## 使い方

```python main.py```

- オプション
    - --model_type: "sequential" or "functional". シーケンシャルモデル、ファンクショナルモデルのどちらを使用するか。（ネットワークは同じ）
    - --batch_size: バッチサイズ
    - --epochs: 学習エポック数
    - --learning_rate: 学習率

# コードの解説
## network.py

モデルを記述している。

### シーケンシャルモデル

`model = Sequential()`でモデルのインスタンスを作成し、そこに、`model.add()`をすることで、
データを追加していく。
1層目のみ`input_shape=`で入力データのサイズを指定する必要があります。このサイズにはバッチサイズは含まない。
順番は, (横, 縦, チャネル) （設定で変更可)

```
def ConvNetSequential(input_shape, n_output):
    model = Sequential()

    # CNN model
    model.add(Conv2D(32, kernel_size=5, strides=1, padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=5, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(n_output))
    model.add(Activation('softmax'))
```

### ファンクショナルモデル

シーケンシャルモデルとは異なり、`x = Conv2D()(x)` というような形で各層の出力を、
次に繋げたい層の引数に入力する。
そして、最後に`Model(inputs=inputs, outputs=output)`で入力と出力を指定する。
シーケンシャルモデルに比べて自由度が高い。

```
def ConvNetFunctional(input_shape, n_output):
    # This returns a tensor
    inputs = Input(shape=input_shape)

    # a layer instance is callable on a tensor, and returns a tensor

    x = Conv2D(32, kernel_size=5, strides=1, padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=5, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    output = Dense(n_output, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=output)
```

## main.py
### 引数を設定。
"-"をつけるとオプション引数になる。
`type`で型を指定。`default`でデフォルトの値を指定。`dest`で参照するときの名前を指定。

```
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="sequential", dest="model_type")
    parser.add_argument("--batch_size", type=int, default=128, dest="batch_size")
    parser.add_argument("--epochs", type=int, default=12, dest="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, dest="learning_rate")
    return parser.parse_args()
```

### データのロード

kerasの関数を使用し、データをロードする。

```
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

### Reshape

kerasでは (バッチサイズ, 横, 縦, チャネル) のshapeのndarrayを入力する。
しかし, modelの最初のレイヤーの引数にはバッチサイズを省くので, `input_shap = (横, 縦, チャネル)`にする。

```
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
```

### 正規化

データを0-1の範囲に抑えることで、早く収束したり、汎化性能が上がることがある。

```
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
```

### ラベルの変換

kerasでは, Loss関数に`keras.losses.categorical_crossentropy`を使用する場合、
ラベルはone-hot形式で与える.
one-hot形式の例) 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] (クラス数が10の場合)

```
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
```

### 最適化関数とLoss関数の設定

model.compileで 最適化関数と, Loss関数を設定し、Loss以外に計算したいメトリクスも設定する。

```
    # set Optimizer and loss function
    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
```

### 学習

model.fit()で学習。しかし、この関数では、データセット全体がメモリに乗る場合にしか使えない。

```
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
```

### 評価

model.evaluate()で与えられたデータに対して、Lossとmodel.compileの時に与えられたメトリクスを計算する。
返り値は, [Loss, メトリクス1, メトリクス2]

```
score = model.evaluate(x_test, y_test, verbose=0)
```
