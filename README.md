
## 深層学習(CNN)と深層距離学習(ArcFace.MagFace)を対象としたプログラムです．

ArcFace
 <https://arxiv.org/abs/1801.07698>
 
MagFace
 <https://arxiv.org/abs/2103.06627>


# Features
リアルタイムで通過した人物がだれかを識別することができる個人認証システムの
画像分類のシステム

深層距離学習を導入することで未知のデータにも対応可能になっています

### 深層距離学習
CNNを特徴量抽出器として使用し出力結果に対して類似度を計算することで
未知データにも対応することを目的としています



実装はGoogle colab上で実装
TensorFlowです


6分割交差検証を行います
main関数内のparse_args内のリスト内で各種パラメーターを一括管理しています

# Requirement
* tensorflow 3.x
* Keras


# Installation
```bash
None
```
 # Author
 
* Kazuki Mori
* Graduate School Student
