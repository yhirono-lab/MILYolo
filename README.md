# Multi-label classification using Multi-scale Domain-adversarial Multiple Instance Learning CNN

このソースコードはデータとしてWSIから切り出したパッチを保存せず，svsファイルをそのまま使用するためのマルチインスタンスCNNアルゴリズムです．

## Environmental Requirement
We confirmed that the source code was running with the following environment.

- python 3.6
- numpy 1.18.1
- opencv-python 4.2.0.32
- pillow 6.1.0
- pytorch 1.4.0
- CUDA 10
- NVIDIA Quadro RTX 5000

## Structure of directories and experimental dataset

We assume the following structure of directories.
ファイルのパスは適宜書き換えてください．

```
root/
　　├ Source/
　　└ Data/
```

Each directory has the following subdirectories and files.

```
Source/
　　├ dataloader_svs.py
　　├ dataset_kurume.py
　　├ eval.py
　　├ export_pos.py
　　├ MIL_test.py
　　├ MIL_train.py
　　└ model.py
```

'dataset_kurume.py'は自身で使用するデータセットに合わせて内容を書き換えてください（クラス数など含め）．

そのまま各プログラムを動かすとディレクトリ'/train_log', '/test_result', '/model_params'などが同じ階層に自動的に生成されます．

An example of structure of directory 'Data' is as follows.
```
Data/
　　├ svs/
　　| 　├ ML_000001.svs
　　|　 ├ ⁝
　　|　 └ ML_999999.svs
　　├ csv
　　|　 ├ ML_000001.csv
　　|　 ├ ⁝
　　|　 └ ML_999999.csv　　　
　　├ thumb_s/
　　└ thumb/
```

最初に用意した'/svs'以外は'export_pos.py'の実行時に自動生成されます．

## Usage
### Single-scale DA-MIL
シングルスケールの学習は以下のコードで実行可能です．

    $ python MIL_train.py 123 4

各引数は以下の通り．
- 第1引数: Trainingデータグループ
- 第2引数: Validationデータグループ

コード内のパラメータは以下の通り．
- 'EPOCHS': 学習エポック数
- 'bag_size': バッグあたりのパッチ画像数
- 'bag_num': 1症例から作成するバッグの最大数．

GPU数=1として実行すれば動作してほしいのですが，エラーが出る場合はMIL_test.pyを参考にしてシングルGPU用に修正してください．

テストも同様のコードで実行可能です．

    $ python MIL_test.py 123 5

引数は学習時と同様です．
- 1行目: Trainingデータグループ(使用する学習モデル)
- 2行目: Testingデータグループ

---

Last update: Apr 28, 2021
