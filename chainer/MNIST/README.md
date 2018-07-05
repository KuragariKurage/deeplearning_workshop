chainerでMNIST分類やってみる
===

## 使い方
実行: 
```
python main.py --batch_size [BATCH_SIZE] -- epoch [EPOCH]--learning_rate [LEANING_RATE] --gpu_idx [GPU_IDX]
```

- オプション
  - `--batch_size`: バッチサイズ (デフォルト 100)

  - `--epoch`: エポック数 (デフォルト 10)

  - `--learning_rate`: 学習率 (デフォルト 1e-08)

  - `--gpu_idx`: 使用するGPU (デフォルト -1 (使用しない))


## コード
- **network.py**
```
class Model(Chain):
    def __init__(self, channel=1, c1=32, c2=64, f1=1024, n_out=10, filter_size1=3, filter_size2=3):
        super(Model, self).__init__(
            conv1=L.Convolution2D(channel,c1,filter_size1),
            conv2=L.Convolution2D(c1, c2, filter_size2),
            l1=L.Linear(None, f1),
            l2=L.Linear(None, n_out)
        )

    def __call__(self, x):
        x = x.reshape((len(x), 1, 28, 28))
        ##forward
        h=F.relu(self.conv1(x))
        h=F.max_pooling_2d(h,2)
        h=F.relu(self.conv2(h))
        h=F.max_pooling_2d(h,2)
        h=F.dropout(F.relu(self.l1(h)))
        y=self.l2(h)

        return y
 ```
 こんなかんじ (後述のtrainerで出力)
 ![accuracy](./mnist_result/cg.png)
 
 
- **main.py**

もともとchainerでMNISTのデータセットが提供されてる。
```
    train, test = chainer.datasets.get_mnist()
    
    train_iter = iterators.SerialIterator(train, minibatch_size)
    test_iter = iterators.SerialIterator(test, minibatch_size, False, False)

```

いろいろ準備。
```
    model = Model()

    if args.gpu_idx>=0:
        chainer.cuda.get_device(args.gpu_idx).use()
        model.to_gpu(args.gpu_idx)


    model=L.Classifier(model)

    optimizer=optimizers.Adam(eps=args.learning_rate)
    optimizer.setup(model)
    
    updater=training.updaters.StandardUpdater(train_iter, optimizer, device=args.gpu_idx)

    max_epoch=args.epoch
```
学習にはtrainerを使います。
```
    trainer=training.Trainer(updater, (max_epoch, 'epoch'),out='mnist_result')
    ...
    trainer.run()
```
extensionでいろいろできる。
```
    #save the log
    trainer.extend(extensions.LogReport())

    #save the model per 5 epoch
    trainer.extend(extensions.snapshot(), trigger=(5, 'epoch'))
    
    #validation
    trainer.extend(extensions.Evaluator(test_iter, model,device=args.gpu_idx))

    #print the log
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))	

    #save the glaph
    trainer.extend(extensions.dump_graph(root_name="main/loss", out_name="cg.dot"))	

    #save the loss glaph
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    #save the accuracy glaph
    trainer.extend(
        extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))

    #print the progress bar
    trainer.extend(extensions.ProgressBar())
```
PlotReportで出てきたグラフ

![loss](./mnist_result/loss.png)
![accuracy](./mnist_result/accuracy.png)


