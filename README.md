# 頭部姿勢推定
## 開発環境
Python 3.6.5  
Keras 2.2.4  
Tensorflow-gpu 1.12.0

## 準備   
### データセットの配置
訓練データとテストデータは、このレポジトリと同じディレクトリに'dataset'ファオルダを作成して、そこに配置する。そして以下のコマンドを実行する。
```
python only_for_mat.py
```
データセットを機械学習で読み込ませるためにフォルダ`data_maker`にある`generate_{軸名}.py`の各関数を実行して、フォルダを分割する。

## 使い方
頭の3つの回転軸ごとに、それぞれのモデルを作成する。回転量の推定には最小2乗法を用いる
### モデルの訓練
```
python train_by_vgg.py --num_epochs [エポック数]\
    --direction [軸(yaw, roll, pitch)]\
    --output_folder [モデルの出力先のファルダ名(数字(1,2...)が好ましい)]
```
### テスト画像での評価
評価のための準備
```
python prepare_for_mls.py --direction [軸(yaw, roll, pitch) --output_folder [モデルの保存先のファルダ名]
```
モデルの評価
```
python test_mode.py --direction [軸(yaw, roll, pitch)]  --output_folder [モデルの保存先のファルダ名]
```
