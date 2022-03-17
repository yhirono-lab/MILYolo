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
必要なデータセット
```
画像データ
root/
　　├ Raw/Kurume_Dataset/svs
　　└ Dataset/Kurume_Dataset/hirono

決定木データ
../KurumeTree
    ├ kurume_tree
    |   ├ 1st
    |   |   ├ Simple   
    |   |   └ Full
    |   |       ├ tree/                     #決定木の図の保存
    |   |       ├ ...  
    |   |       └ unu_depthX/leafs_data     #各深さにおける各葉に分類されたデータセット
    |   |                                   #MILのデータ読み込みに使う
    |   ├ 2nd        
    |   └ 3rd        
    └ normal_tree
        └ 同上
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

プログラムの実行順序と依存関係(run_train.pyで一括実行)
```
Source/
　　├ 1.train.py
　　| 　├ utils.py
　　| 　├ dataset_kurume.py
　　|　 ├ dataloader_svs.py
　　|　 └ make_model.py
　　|　     └ model_yolo.py
　　|　         └ yolo.py
　　|　 
　　├ 2.test.py
　　| 　├ utils.py
　　| 　├ dataset_kurume.py
　　|　 ├ dataloader_svs.py
　　|　 └ make_model.py
　　|　     └ model_yolo.py
　　|　         └ yolo.py　
　　|　 
　　├ 3.result_analyse.py
　　| 　└ utils.py
　　|　 
　　└ 4.draw_heatmap.py
　　  　└ utils.py
```

各プログラムには実行時にパラメータの入力が必要
run_train.pyで一括実行する際は，train.py,test.pyの*の付いたパラメータに注意
```
train               データのクロスバリデーションの番号を指定 例：123
--depth             決定木の深さを指定 例：1
--leaf              決定木の指定した深さの葉の指定 例：01
--yolo_ver          学習したyoloの重みを選択(multistageの時はbest固定) 例：1
--data              データ数の選択 (1st, 2nd, 3rd)
--mag               拡大率の選択(40x以外はデータセットがないから不可) 例：40x
--model             MILの特徴抽出器の選択 (vgg16 or vgg11)
--dropout           (flag)ドロップアウト層の有無
--multistage        multistageにおける階層数の入力 例:1
--detect_obj        YOLOAMILにおけるYOLOの特徴ベクトルのサイズ(defaultは3087) 例：50
*--epoch            epoch数の指定(1epoch 40分目安) 例：10
*--batch_size       バッチサイズの指定(singlestageは4，multistageは3までいけたはず) 例：3
--name              データセット名の指定(normal_tree,subtypeの時に有効)
                        * Simple : 例 DLBCL, FL
                        * Full : 例 DLBCL-GCB, FL-grade1
-c, --classify_mode 識別する決定木の種類の選択 
                        * normal_tree : エントロピーによる決定木で識別
                        * kurume_tree : 免疫染色による決定木で識別
                        * subtype : サブタイプ名別で識別(5種類)
-l, --loss_mode     損失関数の選択
                        * CE : Cross Entropy Loss
                        * ICE : Inversed Cross Entropy Loss (重み付けCE)
                        * focal : Focal Loss
                        * focal-weight : Weighted Focal Loss (重み付けFocal Loss)
                        * LDAM : LDAM Loss
--lr                学習率の指定 例：0.00005
*--weight_decay     weight_decayの指定 例：0.0001
*--momentum         momentumの指定 例：0.9
-C, --constant      LDAM損失使用時のパラメータ指定(0～0.5くらい) 例：0.2
-g, --gamma         focal損失使用時のパラメータ指定 例：1.0
-a, --augmentation  (flag)回転・反転によるaugmentationの有無
*-r, --restart      (flag)再学習するか否か(上手くできているか不明)
--fc                (flag)ラベル予測部分のみを学習するか否か
--reduce            (flag)データ数が多い時に多すぎるラベルを減らすか否か
*--device           使用するGPU番号の選択 例：0,1,2,3
```
*注意1  
--restartは上手く動かない可能性有  
model_mil.py,model_yolo.pyでrestart時のパラメータを選択するが，best.ptとckpt.ptのせいで最後のepochのパラメータになってないかもしれない  
*注意2
画像svsファイル名の変更や，パラメータの名前の変更によってエラーが発生する可能性あるかもしれないです  

各プログラムを実行するとパラメータごとにruns/に結果が保存される．
同一の設定で実験を行った場合，自動的に番号付けされて別のディレクトリに保存される
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

## 未報告次項
* YOLOの特徴量の制限(自主的な改良)
    そのままYOLOの特徴量を用いると3087次元であり，対象の細胞核が存在していないときは検出確率0が大半をしめる特徴ベクトルとなる．  
    一方で，vggによる特徴量は512次元である．  
    vggの特徴量に対してYOLOの特徴量が多すぎると識別に悪影響がでるかもしれないと思い，YOLOの特徴量を検出確率の上位X個によるX次元の特徴量に削減して実験できるようにした．  
    結果としては，そこまで精度が変わらなかった(むしろ下がった？)

* YOLOのマルチステージ学習(先生提案の改良)
    決定木のB細胞性vsMETAの分類ではDLBCLやFLの識別性能が非常に良いが，一つ上の階層での識別(T細胞性vsその他)ではDLBCLやFLなどの分類がいまいち
    そこで，下位層のYOLOの特徴を用いることができれば，上位層の識別性能が向上すると考えた．
    結果としては，そこまで精度が変わらなかった(むしろ下がった？)

#### ファイル名の変更の対応関係
データセットのバージョン
```
(修飾無し) → 1st  
add_ → 2nd  
New : 3rd  (実験したことのないのでエラーに注意)
```

決定木の名前
```
leaf → normal_tree
new_tree → kurume_tree
subtype → subtype
```