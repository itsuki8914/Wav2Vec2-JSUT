# Wav2Vec2-JSUT
本リポジトリでは、[Wav2Vec2による日本語音声認識を試してみる](https://zenn.dev/itsuki9180/articles/f0f5e409a9c808)の実装を公開及び管理しています。本ソースコードはMITライセンスで公開しています。またこのリポジトリは、インプレス社機械学習実践シリーズの[Pythonで学ぶ音声認識（機械学習実践シリーズ）](https://book.impress.co.jp/books/1120101083)より多くのコードを引用させていただいています。この場を借りて感謝申し上げます。

## ソースコード
- 00_prepare.ipynb
  JSUTコーパスのダウンロードとサンプリング周波数のリサンプルを行います。
- 01_preprocess.ipynb
  ラベルデータから必要な情報を抜き出し、csvに出力します。
- 02_Wav2Vec2_tiny_train.ipynb
  Wav2Vec2の学習を行います。
- 03_inference.ipynb 
  完成したモデルの評価を行います。
- omake
  おまけです。詳細は省きます。
