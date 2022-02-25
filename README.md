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

必要なソースコード

```
Source/
　　├ dataloader_svs.py     #データローダーの作成のプログラム
　　├ dataset_kurume.py     #データセットの作成のプログラム
　　├ draw_heatmap.py       #Attentionの可視化のプログラム
　　├ make_model.py         #model作成のプログラム
　　├ model_yolo.py         #YOLOAMILで使用するモデルのプログラム
　　├ result_analyse.py     #Lossのグラフやテストのスコアを計算するプログラム
　　├ run_train.py          #必要なプログラムを一括実行するプログラム
　　├ test.py               #テスト用プログラム
　　├ train.py              #訓練用プログラム
　　├ utils.py              #汎用的な関数のプログラム
　　└ yolo.py               #YOLOAMILのYOLOモデル部分のプログラム
```

プログラムの実行順序と依存関係
```
Source/
　　├ train.py
　　| 　├ utils.py
　　| 　├ dataset_kurume.py
　　|　 ├ dataloader_svs.py
　　|　 └ make_model.py
　　|　     └ model_yolo.py
　　|　         └ yolo.py
　　|　 
　　├ test.py
　　| 　├ utils.py
　　| 　├ dataset_kurume.py
　　|　 ├ dataloader_svs.py
　　|　 └ make_model.py
　　|　     └ model_yolo.py
　　|　         └ yolo.py　
　　|　 
　　├ result_analyse.py
　　| 　└ utils.py
　　|　 
　　└ draw_heatmap.py
　　  　└ utils.py
```
  
各プログラムには実行時にパラメータの入力が必要
run_train.pyで一括実行する際は，train.py,test.pyの*の付いたパラメータに注意
```
train               データの5分割の番号を指定 例：123
--depth             決定木の深さを指定 例：1
--leaf              決定木の指定した深さの葉の指定 例：01
--yolo_ver          学習したyoloの重みを選択(multistageの時はbest固定) 例：1
--data              データ数の選択(基本はadd) 例：add
--mag               拡大率の選択(40x以外はデータセットがないから不可) 例：40x
--model             MILの特徴抽出器の選択 例：vgg16
--dropout           (flag)ドロップアウト層の有無
--multistage        multistageにおける階層数の入力 例:1
--detect_obj        YOLOAMILにおけるYOLOの特徴ベクトルのサイズ(defaultは3087) 例：50
*--epoch            epoch数の指定(1epoch 40分目安) 例：10
*--batch_size       バッチサイズの指定(singlestageは4，multistageは3までいけたはず) 例：3
--name              データセット名の指定(unu_treeの時に有効) 例：Simple
-c, --classify_mode 識別する決定木の種類の選択 例：new_tree
-l, --loss_mode     損失関数の選択 例：ICE
--lr                学習率の指定 例：0.00005
*--weight_decay     weight_decayの指定 例：0.0001
*--momentum         momentumの指定 例：0.9
-C, --constant      LDAM損失使用時のパラメータ設定(0～0.5くらい) 例：0.2
-g, --gamma         focal損失使用時のパラメータ設定 例：1.0
-a, --augmentation  (flag)回転・反転によるaugmentationの有無
*-r, --restart      (flag)再学習するか否か(上手くできているか不明)
--fc                (flag)ラベル予測部分のみを学習するか否か
--reduce            (flag)データ数addの時に多すぎるラベルを減らすか否か
*--device           使用するGPU番号の選択 例：0,1,2,3
```
*注意1  
--restartは上手く動かない可能性有  
model_mil.py,model_yolo.pyでrestart時のパラメータを選択するが，best.ptとckpt.ptのせいで最後のepochのパラメータになってないかもしれない  
*注意2  
--mil_mode amilはモデルの出力のエラーで上手く動かないかもしれない．

各プログラムを実行するとパラメータごとにruns/に結果が保存される．
同一の設定で実験を行った場合，
```
runs/
　　└ パラメータ別のディレクトリ1/
　　 　 └ パラメータ別のディレクトリ2/
　　 　     ├ attention_map/        Attention-weightのカラーマップを保存
　　 　     ├ attention_patch/      ラベル別にAttention-weightが高いパッチを保存
　　 　     ├ total_result/graph/   学習時のグラフやテストのスコアを保存
　　 　     ├ trainXXX/
　　  　    | 　　├ graphs/         各学習のグラフを保存
　　  　    | 　　├ graphs/         各学習のグラフを保存
　　      　|　 　├ model_params/   
　 　   　  | 　　|　 　├ ...epoch-x.pt     各epochにおけるモデルのパラメータ
　 　   　  | 　　|　 　├ best.pt           valid_loss最小になったモデルのパラメータ
　 　   　  | 　　|　 　└ ckpt.pt           valid_loss最小になったoptimizerのパラメータ
　　      　|　 　└ test_result/
　 　   　  | 　　|　 　├ bbox_coords/              YOLOによって検出された細胞核の情報を保存
　 　   　  | 　　|　 　└ test...train-XXX_best.csv テストデータにおける実験結果の保存
　 　   　  | 　　└ log...train-123.csv　   訓練時における実験結果の保存
　　 　     └ best_exps.txt/        ベストスコアの実験のメモ
```

