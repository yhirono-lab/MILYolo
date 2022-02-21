# Attention-based Multiple Instance Learning with YOLO

YOLOを用いたAMILのプログラム．
パッチ画像は保存せず，/Raw/Kurume_Dataset/svs に保存されたsvsファイルから逐次パッチ画像を作成する．

## 実行環境
ライブラリバージョン
- python 3.8.10
- numpy 1.21.2
- opencv-python 4.5.3.56
- openslide-python 1.1.2
- pillow 8.3.1
- torch 1.9.0
- torchvision 0.10.0

使用したマシン
- マシン noah
- CUDA Version 11.4
- Driver Version 470.86

## OpenSlide のインストール
OpenSlideというライブラリをマシンにインストールしないとパッチ画像の作成ができない．
(通常であれば，入っている環境に設定してくれていると思う)
python用のライブラリをダウンロード
    ```
    apt-get install python-openslide
    apt-get install openslide-tools
    pip3 install openslide-python
    ```

## ファイル構造
久留米データセット
```
root/
　　├ Raw/Kurume_Dataset/
　　└ Dataset/Kurume_Dataset/
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
