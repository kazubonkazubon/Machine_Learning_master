import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.preprocessing.image import array_to_img, img_to_array, load_img
import tensorflow.keras.backend as K
import argparse
import configparser
import sys
from keras.applications.vgg16 import VGG16
import glob
import os
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization,GlobalAveragePooling2D
from keras.models import Sequential, load_model, Model
from sklearn.svm import SVC
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
sys.path.append("/content/drive/MyDrive/Arc_face")
weight_decay = 1e-4

from metrics import *

class L2ConstrainLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(L2ConstrainLayer, self).__init__()
        self.alpha = tf.Variable(30.)
    def call(self, inputs):
        #about l2_normalize https://www.tensorflow.org/api_docs/python/tf/keras/backend/l2_normalize?hl=ja
        return K.l2_normalize(inputs, axis=1) * self.alpha

def step_decay(epoch):
    """
    input 
    epoch (type int)　:一定のepochを超えた時にlearning lateを減少させる
  
    return 
      lr    (type int)  :減少させた結果のlearning　lateの大きさ
    """
    lr = 0.001
    if(epoch >= 100):
        lr/=5
    if(epoch>=140):
        lr/=2
    return lr

def get_labels():
  """
  input
    null

  return 
    labels  (type list)  :labels.txtに記述されている分類対象をlistで返す
  
  """
  f = open("/content/model/labels.txt", 'r')
  labels=[]
  datalist = f.readlines()
  for z in datalist:
    labels.append(z[:-1])
  return labels

def get_filter_num():
  """
  input 
    null

  return 
    CNNでの入力層の画像サイズのピクセル数を返す

  """
  f = open('/content/input_size.txt', 'r')
  num=int(f.read())
  return num

"""
ここから下はCNNのmodel architectureの定義
input 
  activation     (type string)        :modelで用いている最適化関数　　多くのライブラリでは動的に定義しますが，現状全てadamで固定
  mid_units      (type int)           :最終結合層直前の中間層のニューロンの数　default:512
  dropout_rate   (type double)        :drop out層でのdrop outの割合　　defaultは要config.txt参照のこと
  dropout_rate_2 (type double)        :以下同文
  X_train        (type numpy.darray)  :trainデータの画像データ側　　入力サイズは(画像数,64,64,color) colorはRGBだと3 Gray_scaleでは1となる点に注意
  LEARNING_LATE  (type double)        :train開始時に決定したLearning Lateの大きさ　一般にこの大きさが大きくなると学習が高速になるが収束はしにくくなる　逆もしかり

return 
  model (type tensorflow.Functional) :定義されたmodelを返す
"""

from keras import backend as K
import numpy as np
import tensorflow as tf

class ArcFace(Layer):
    #class 数変える時は変えるんご
    def __init__(self, n_classes=get_class_num(), s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)
    def get_config(self):
        #書き換えるんご
        config = {
            "n_classes" : self.n_classes,
            "s" : self.s,
            "m" : self.m,
            "regularizer" : self.regularizer,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items())) 

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)
        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)



def mk_models_NO1(activation,   mid_units, dropout_rate,dropout_rate_2,X_train,LEARNING_RATE):
    """
    RGB画像を対象に分類を行います
    入力サイズは基本(image_num,64,64,3) 
    畳み込みは2回
    """    
    filter_num=get_filter_num()
    labels=get_labels()
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=(filter_num, filter_num, 3),name="conv_1"))
    model.add(Activation('relu',name="relu_1"))
    model.add(Conv2D(64, (3, 3),name="conv_2"))
    model.add(Activation('relu',name="relu_2"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(labels)))
    model.add(Activation('softmax',name="softMax_func"))
    #sgd = optimizers.SGD(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
    return model

def mk_models_NO2(activation,mid_units,dropout_rate,dropout_rate_2,dropout_rate_3,X_train,LEARNING_RATE):
    """
    RGB画像を対象に分類を行います
    入力サイズは基本(image_num,64,64,3) 
    畳み込みは4回
    """
    model = Sequential()
    labels=get_labels()
    model.add(Conv2D(32, (3, 3), padding='same', 
                input_shape=(64, 64, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(dropout_rate_2))
    model.add(Flatten())
    model.add(Dense(mid_units))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate_3))
    model.add(Dense(len(labels)))
    model.add(Activation('sigmoid'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

def mk_models_NO1_GRAY_SCALE(activation,   mid_units, dropout_rate,dropout_rate_2,X_train,LEARNING_RATE):
    """
    Glay_Scale画像を対象に分類を行います
    入力サイズは基本(image_num,64,64,1) 
    畳み込みは2回
    """
    labels=get_labels()
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=(64, 64, 1),name="conv_1"))
    model.add(Activation('relu',name="relu_1"))
    model.add(Conv2D(64, (3, 3),name="conv_2"))
    model.add(Activation('relu',name="relu_2"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(mid_units))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate_2))
    model.add(Dense(len(labels)))
    model.add(Activation('softmax'))
    #sgd = optimizers.SGD(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
    return model

def mk_models_NO2_GRAY_SCALE(activation,mid_units,dropout_rate,dropout_rate_2,dropout_rate_3,X_train,LEARNING_RATE):
    """
    Glay_Scale画像を対象に分類を行います
    入力サイズは基本(image_num,64,64,1) 
    畳み込みは4回
    """
    model = Sequential()
    labels=get_labels()
    model.add(Conv2D(32, (3, 3), padding='same', 
                input_shape=(64, 64, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(dropout_rate_2))
    model.add(Flatten())
    model.add(Dense(mid_units))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate_3))
    model.add(Dense(len(labels)))
    model.add(Activation('sigmoid'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

def get_vgg16_model():
    model_path = "./models/vgg16.h5py"
    if not os.path.exists(model_path):
        input_tensor = Input(shape=(224,224,3))
        # 出力層側の全結合層３つをモデルから省く
        model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
        model.save(model_path) # 毎回ダウンロードすると重いので、ダウンロードしたら保存する
    else:
        model = load_model(model_path)
    return model

def create_last_conv2d_fine_tuning(classes,fine_tuning):
    # vgg16モデルを作る
    vgg16_model = get_vgg16_model()
    input_tensor = Input(shape=(224,224,3))
    x = vgg16_model.output
    """
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    """
    n_categories=len(get_labels())
    yinput = Input(shape=(n_categories,))
    hidden = GlobalAveragePooling2D()(x)
    x = Arcfacelayer(5, 30, 0.05)([hidden,yinput])
    prediction = Activation('softmax')(x)

    base_model=vgg16_model

    model = Model(inputs=[base_model.input,yinput],outputs=prediction)

    
    if fine_tuning==True:
      print("finetuningに伴い　パラメーターの凍結を行います")
      # 最後の畳み込み層より前の層の再学習を防止
      for layer in model.layers[:15]: 
        layer.trainable = False
    else :
      print("全てのパラメータをチューニングします")
    #optimizerss = keras.optimizers.Adam(learning_rate=0.0001)

    aa=tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=aa, loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def vgg2_color_arcface(args):
  class_num=get_class_num()
  input = Input(shape=(64, 64, 3))
  y = Input(shape=(class_num,))
  x=Conv2D(32,(3,3),padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(weight_decay))(input)
  #x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x=Conv2D(64,(3,3),padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(weight_decay))(x)
  #x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x=Dropout(0.5)(x)
  #x=Flatten()(x)
  #x = Dense(args.num_features, kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(weight_decay))(x)
  x=Dropout(0.25)(x)
  x = Activation('relu')(x)
  #x=Batchnormalization()(x)
  class_num=get_class_num()
  x= GlobalAveragePooling2D()(x)
  x = ArcFace(class_num, regularizer=regularizers.l2(weight_decay))([x, y])
  prediction=Activation('softmax')(x)
  model=Model([input,y],prediction)
  return model



def mk_models_VGG16_ArcFace_fine_tuning():
  # vgg16モデルを作る
  base_model=tf.keras.applications.vgg16.VGG16(input_shape=(224,224,3),
                       weights='imagenet',
                       include_top=False)
  n_categories=len(glob.glob("/content/based_image_data/*"))

  yinput = Input(shape=(n_categories,))
  

  x = base_model.output
  yinput = Input(shape=(n_categories,))
  # stock hidden model
  hidden = tf.keras.layers.GlobalAveragePooling2D()(x)
  # stock Feature extraction
  #x = Dropout(0.5)(hidden)
  x = Arcfacelayer(n_categories, 30, 0.05)([hidden,yinput])
  #x = Arcfacelayer(5, 30, 0.05)([hidden,yinput])
  #x = Dense(1024,activation='relu')(x)
  prediction = Activation('softmax')(x)
  model = Model(inputs=[base_model.input,yinput],outputs=prediction) 

  fine_tuning=True
  #model = Model(inputs=vgg16_model.input, outputs=predictions)
  
  
  
  if fine_tuning==True:
    print("finetuningに伴い　パラメーターの凍結を行います")
    # 最後の畳み込み層より前の層の再学習を防止
    for layer in model.layers[:15]: 
      layer.trainable = False
  else :
    print("全てのパラメータをチューニングします")
    #optimizerss = keras.optimizers.Adam(learning_rate=0.0001)
  aa=tf.keras.optimizers.Adam(learning_rate=0.0001)
  model.compile(optimizer=aa, loss='categorical_crossentropy', metrics=['accuracy'])



def VGG_16_trans_MLP(args):
  base_model = tf.keras.applications.vgg16.VGG16(include_top=False,
                                               input_shape=(224,224,3),
                                               weights='imagenet')
  base_model.trainable = False
  x = tf.keras.layers.Flatten()(base_model.output)
  x = tf.keras.layers.Dense(4096, activation='relu')(x)
  x = tf.keras.layers.Dense(4096, activation='relu')(x)
  transfer_learning_inputs = base_model.inputs
  image_class_num=len(glob.glob("/content/based_image_data/*"))
  image_class_names=get_labels()
  transfer_learning_prediction = tf.keras.layers.Dense(
                                 image_class_num, activation='softmax')(x)
  transfer_learning_model = tf.keras.Model(inputs=transfer_learning_inputs,
                                         outputs=transfer_learning_prediction)
  aa=tf.keras.optimizers.Adam(learning_rate=0.0001)
  transfer_learning_model.compile(optimizer=aa, loss='categorical_crossentropy', metrics=['accuracy'])
  return transfer_learning_model




def VGG_16_trans_MLP_before1(args):
  base_model = tf.keras.applications.vgg16.VGG16(include_top=False,
                                               input_shape=(224,224,3),
                                               weights='imagenet')
  base_model.trainable = False
  x = tf.keras.layers.Flatten()(base_model.output)
  x = tf.keras.layers.Dense(4096, activation='relu')(x)
  transfer_learning_inputs = base_model.inputs
  image_class_num=len(glob.glob("/content/based_image_data/*"))
  image_class_names=get_labels()
  transfer_learning_prediction = tf.keras.layers.Dense(
                                 image_class_num, activation='softmax')(x)
  transfer_learning_model = tf.keras.Model(inputs=transfer_learning_inputs,
                                         outputs=transfer_learning_prediction)
  aa=tf.keras.optimizers.Adam(learning_rate=0.0001)
  transfer_learning_model.compile(optimizer=aa, loss='categorical_crossentropy', metrics=['accuracy'])
  return transfer_learning_model


def VGG_16_trans_MLP_before2(args):
  base_model = tf.keras.applications.vgg16.VGG16(include_top=False,
                                               input_shape=(224,224,3),
                                               weights='imagenet')
  base_model.trainable = False
  x = tf.keras.layers.Flatten()(base_model.output)
  transfer_learning_inputs = base_model.inputs
  image_class_num=len(glob.glob("/content/based_image_data/*"))
  image_class_names=get_labels()
  transfer_learning_prediction = tf.keras.layers.Dense(
                                 image_class_num, activation='softmax')(x)
  transfer_learning_model = tf.keras.Model(inputs=transfer_learning_inputs,
                                         outputs=transfer_learning_prediction)
  aa=tf.keras.optimizers.Adam(learning_rate=0.0001)
  transfer_learning_model.compile(optimizer=aa, loss='categorical_crossentropy', metrics=['accuracy'])
  return transfer_learning_model



def MobileNetV2_trans_MLP():
  base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,
                                               input_shape=(224,224,3),
                                               weights='imagenet')
  base_model.trainable = False
  x = GlobalAveragePooling2D()(base_model.output)
  
  transfer_learning_inputs = base_model.inputs
  image_class_num=len(glob.glob("/content/based_image_data/*"))
  image_class_names=get_labels()
  transfer_learning_prediction = tf.keras.layers.Dense(
                                 image_class_num, activation='softmax')(x)
  transfer_learning_model = tf.keras.Model(inputs=transfer_learning_inputs,
                                         outputs=transfer_learning_prediction)
  aa=tf.keras.optimizers.Adam(learning_rate=0.00001)
  transfer_learning_model.compile(optimizer=aa, loss='categorical_crossentropy', metrics=['accuracy'])
  return transfer_learning_model

def ResNet_trans_MLP():
  base_model = ResNet50(weights='imagenet',include_top=False)
  base_model.trainable = False
  x = GlobalAveragePooling2D()(base_model.output)
  
  transfer_learning_inputs = base_model.inputs
  image_class_num=len(glob.glob("/content/based_image_data/*"))
  image_class_names=get_labels()
  transfer_learning_prediction = tf.keras.layers.Dense(
                                 image_class_num, activation='softmax')(x)
  transfer_learning_model = tf.keras.Model(inputs=transfer_learning_inputs,
                                         outputs=transfer_learning_prediction)
  aa=tf.keras.optimizers.Adam(learning_rate=0.00001)
  transfer_learning_model.compile(optimizer=aa, loss='categorical_crossentropy', metrics=['accuracy'])
  return transfer_learning_model

def VGG_16_trans_MLP_fine_tuning():
  image_resize=224
  input_tensor = Input(shape=(image_resize, image_resize, 3))
  vgg16_model = VGG16(
    include_top=False, #全結合層を除外
    weights='imagenet', 
    input_tensor=input_tensor
    )
  for layer in vgg16_model.layers:
    layer.trainable = False
  # 全結合層の構築
  x = vgg16_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='relu')(x)
  x = Dropout(0.5)(x)
  num_classes=len(glob.glob("/content/based_image_data/*"))
  
  predictions = Dense(num_classes, activation="softmax")(x)
  aa=tf.keras.optimizers.Adam(learning_rate=0.00001)
  model = Model(inputs=vgg16_model.input, outputs=predictions)
  model.compile(optimizer=aa, loss='categorical_crossentropy', metrics=['accuracy'])
  # VGG16と構築した全結合層を結合
  return model



def get_model(args,mode,section,X_train):
  config = configparser.ConfigParser()
  config.read('/content/drive/MyDrive/parameters/config.ini')
  mid_units          =int(float(config.get(section,'mid_units')))
  LEARNING_RATE      =args.lr
  dropout_rate       =float(config.get(section,'dropout_rate'))
  dropout_rate_2    =float(config.get(section,'dropout_rate_2'))
  dropout_rate_3    =float(config.get(section,'dropout_rate_3'))
  EPOCHS             =args.epochs
  activation=args.optimizer
  print(args.arch)
  if args.arch=="CNN_NO1" :
    if mode==3:
      model=mk_models_NO1(activation,   mid_units, dropout_rate,dropout_rate_2,X_train,LEARNING_RATE)
    else:
      model=mk_models_NO1_GRAY_SCALE(activation,   mid_units, dropout_rate,dropout_rate_2,X_train,LEARNING_RATE)
  elif args.arch=="CNN_NO2":
    if mode==3:
      model=mk_models_NO2(activation,mid_units,dropout_rate,dropout_rate_2,dropout_rate_3,X_train,LEARNING_RATE)
    if mode==1:
      model=mk_models_NO2_GRAY_SCALE(activation,mid_units,dropout_rate,dropout_rate_2,dropout_rate_3,X_train,LEARNING_RATE)
  elif args.arch=="VGG_16_fine_tuning":
    CLASS_NUM=len(glob.glob("/content/train_data/*"))
    fine_tuning=True
    model=create_last_conv2d_fine_tuning(CLASS_NUM,fine_tuning)
  
  elif args.arch=="VGG_16_ArcFace_fine_tuning":
    if mode==3:
      model=mk_models_VGG16_ArcFace_fine_tuning()
    if mode==1:

      pass

  elif args.arch=="VGG16_ArcFace":

    pass
  elif args.arch=="VGG_16_trans_MLP":
    model=VGG_16_trans_MLP(args)
  elif args.arch=="MobileNetV2_trans_MLP":
    model=MobileNetV2_trans_MLP()
  elif args.arch=="ResNet_trans_MLP":
    model=ResNet_trans_MLP()
  elif args.arch=="VGG_16_trans_SVM":
    model = tf.keras.applications.vgg16.VGG16(include_top=False,
                                               input_shape=(224,224,3),
                                               weights='imagenet',pooling="avg")
    model.trainable = False
  elif args.arch=="MobileNetV2_trans_SVM":
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,
                                               input_shape=(224,224,3),
                                               weights='imagenet',pooling="avg")
    model.trainable = False
  elif args.arch=="InceptionV3_trans_SVM":
    model = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                                               input_shape=(224,224,3),
                                               weights='imagenet',pooling="avg")
    model.trainable = False

  elif args.arch=="VGG_19_trans_SVM":
    model = tf.keras.applications.vgg19.VGG19(include_top=False,
                                               input_shape=(224,224,3),
                                               weights='imagenet',pooling="avg")
    model.trainable = False
  elif args.arch=="ResNet_trans_SVM":
    model = tf.keras.applications.resnet50.ResNet50(include_top=False,input_shape=(224,224,3),weights='imagenet',pooling="avg")
    model.trainable = False
    pass
  elif args.arch=="SVM":
    model=SVC()  
    pass
  elif args.arch=="VGG_16_trans_MLP_fine_tuning":
    model=VGG_16_trans_MLP_fine_tuning()
  else:
    print("ERROR:該当する学習モデルが存在しません")
    sys.exit()
  return  model
