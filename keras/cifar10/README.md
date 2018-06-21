# 使い方

## 概要
CIFAR-10データセットを用いて、10種類の物体を識別するネットワークを作成し、テストも行う。
MNISTとは異なり、ローカルに保存したデータを学習に使用。
浅いネットワークやし、batchNormalization()もしてないので、あまり精度はよくない。
- Train loss 0.2038 acc: 0.9281
- Validation loss 0.7716 acc:0.7872 (Trainingデータの一部を使用)
- Test loss 0.7759 acc: 0.7913

## 使い方

```
python make_dataset.py  # CIFAR-10のデータをdata/imageに保存 & data/にtrain, validation, testのデータセットファイルを作成
python main.py # 学習 & 評価
```

- main.py オプション
    - --train_dataset_path: 学習用データセットへのパス。 "画像のパス,ラベル" の形式
    - --validation_dataset_path: 評価用データセットへのパス。 "画像のパス,ラベル" の形式
    - --test_dataset_path: テスト用データセットへのパス。 "画像のパス,ラベル" の形式
    - --model_type: "sequential" or "functional". シーケンシャルモデル、ファンクショナルモデルのどちらを使用するか。（ネットワークは同じ）
    - --batch_size: バッチサイズ
    - --epochs: 学習エポック数
    - --learning_rate: 学習率

# コードの解説
## make_dataset.py

Cifar-10のデータをダウンロードして画像として保存し、さらに、"画像のパス,ラベル" 形式の
データセットファイルも生成する。
画像は`data/image`ディレクトリに、データセットファイルは`data/`ディレクトリに保存。
正直、画像を保存して、パスをファイルに書き出しているだけである。
`skearn.model_selection.train_test_split()`はデータを学習用とvalidation用に分割する時などに便利。

```
    # split train data to validation data
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train, y_train, test_size=val_rate
    )
```

## network.py

モデルを記述している。
基本的には、MNISTの`network,py`と同様。
異なる点は、`Dropout()`を使用している点。Dropout()の引数はドロップする割合。
ドロップアウトを使用すると、過学習の防止に役立つ。

## data_generator.py

メモリに乗らないような、大量のデータを学習に使用する場合も考慮して実装。
このクラスのインスタンスが作成されると、まず、`__init__()`が呼ばれる。
`__init__()`ではデータセットファイルをパースして、画像のパスとラベルをメンバ変数に保存している。
また、`keras.util.np_utils`を使ってラベルをone-hot形式に変換する。

```
    def __init__(self, dataset_file, num_output, batch_size=32, shuffle=True):
        self.data_path_list = []
        self.label_list = []
        self.batch_size = batch_size
        self.shuffle = shuffle

        for line in open(dataset_file, "r"):
            data_path, label = line.split(",")

            self.data_path_list.append(data_path)
            self.label_list.append(int(label))

        self.label_list = np_utils.to_categorical(self.label_list, num_output)

```

次に、バッチを取り出す関数を実装する。
メモリに乗り切らないようなデータを扱う場合は、バッチごとにデータを読み出す。
`yield`とつけるとジェネレータになり, この行が呼ばれると, 一度引数を返し,
再度この関数が呼ばれた時に続きから実行する。

```
    def get_batch(self):
        while 1:
            if self.shuffle is True:
                self.data_path_list, self.label_list = my_array_operation.shuffle_two_list(self.data_path_list, self.label_list)

            for i in range(0, len(self.data_path_list), self.batch_size):
                # データをロード
                data = []
                for data_path in self.data_path_list[i:i + self.batch_size]:
                    data.append(cv2.imread(data_path, cv2.IMREAD_COLOR))
                label = self.label_list[i:i + self.batch_size]

                # 正規化
                data = np.array(data, dtype=np.float)
                data /= 255

                yield data, np.array(label)
```

後から、1epochあたりのイテレーション数や入力サイズが必要になるので、それを取得する関数も実装。

```
    def get_iter_num(self):
        return int(len(self.data_path_list) / self.batch_size)

    def get_batch_shape(self):
        batch = []
        for data_path in self.data_path_list[0:0 + self.batch_size]:
            batch.append(cv2.imread(data_path, cv2.IMREAD_COLOR))
```

## main.py

先ほど作成した、data_generator.pyのクラス`dataGenerator()`を使用。
training, validation, test 用のdataGenerator()を作成する。

```
    train_dataset = dataGenerator(train_dataset_path, num_classes, batch_size, shuffle=True)
    validation_dataset = dataGenerator(validation_dataset_path, num_classes, batch_size, shuffle=True)
    test_dataset = dataGenerator(test_dataset_path, num_classes, batch_size, shuffle=True)
```

そして、`model.compile()`で最適化関数とLoss関数を設定し、`model.fit_generator()`を使用して、学習。
`model.fit_generator()`には, ジェネレータやリストが使用できるが、今回はジェネレータを使用する。
その場合、1epochあたりのステップ数(イテレーション数)を指定する必要がある。
また、ジェネレータの場合はshuffle引数は意味がないらしいので, 自分でdataGenerator()内に実装。

```
    # set Optimizer and loss function
    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # learning
    model.fit_generator(train_dataset.get_batch(), train_dataset.get_iter_num(), epochs,
                        verbose=1, validation_data=validation_dataset.get_batch(),
                        validation_steps=validation_dataset.get_iter_num(), shuffle=False) # shuffle is implementted in my dataGenerator.
```

テストも同様

```
    score = model.evaluate_generator(test_dataset.get_batch(), steps=test_dataset.get_iter_num(),
                                     verbose=1)
```
