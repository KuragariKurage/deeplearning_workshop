ChainerUIで可視化
===
tensorboardで可視化してみようって話だったけどchainer用の可視化ツールChainerUIが提供されてるのでそっちを使ってみた。


```
pip install chainerui
chainerui db create
chainerui db update
```
で準備完了。

プロジェクトを登録。
```
chainerui project create -d [LOG_DIR] -n [PROJECT_NAME]
```

サーバの起動。
```
chainerui server
```


いろいろな機能があるけどこのサイトが分かりやすかった。
http://t2kasa.sub.jp/chainerui-introduction-and-usage
