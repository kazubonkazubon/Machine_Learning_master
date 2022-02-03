import os
import shutil
import argparse
import sys
import glob
import cv2
import datetime
import matplotlib.pyplot as plt
from natsort import natsorted


import module_1 as func
import model_define as m_define 
import Data_argumation_operation as argumation_operation
import preds_data_func as Preds_Data_Func
import make_dirs_func as make_data
import tensorflow as tf
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.python.keras import optimizers
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import itertools
import seaborn as sns
import shutil
import os
import cv2
from matplotlib.cm import get_cmap
#from keras.utils import plot_model
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import ShuffleSplit
import subprocess
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import model_from_json
import glob
import configparser
import datetime
import pytz
import openpyxl
import re
from keras.callbacks import Callback
import create_model
from keras.callbacks import EarlyStopping
import keras
from sklearn.model_selection import ShuffleSplit
from keras import regularizers
from sklearn.svm import SVC

sys.path.append("/content/drive/MyDrive/functions/")
sys.path.append("/content/drive/MyDrive/Arc_face/")
import make_dirs_func as MAKE_DIR
import scheduler
import classified_others
weight_decay = 1e-4


import math
from keras.callbacks import Callback
from keras import backend as K
from keras import optimizers

class CosineAnnealingScheduler(Callback):
    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

def prcess(file_path,num_data):
  if str(num_data) in file_path and not "hdf5" in file_path:
    if not "tensorboard" in file_path:
      print(file_path)
      if os.path.exists(file_path):
        os.remove(file_path)
  
def recursive_file_check(path,num_data):
    if os.path.isdir(path):
        # directoryだったら中のファイルに対して再帰的にこの関数を実行
        files = os.listdir(path)
        for file in files:
            recursive_file_check(path + "/" + file,num_data)
    else:
        # fileだったら処理
        prcess(path,num_data)
def kill_dir(a):
  for i in a:
    shutil.rmtree(i)

def kill_data():
  x=glob.glob("/content/drive/MyDrive/models/*/tensorBoard/*")
  kill_dir(x)
  x=glob.glob("/content/drive/MyDrive/models/*/model_datas")
  kill_dir(x)
  x=glob.glob("/content/drive/MyDrive/models/*/acc")
  kill_dir(x)
  x=glob.glob("/content/drive/MyDrive/models/*/loss")
  kill_dir(x)
  x=glob.glob("/content/drive/MyDrive/models/*/val_acc")
  kill_dir(x)
  x=glob.glob("/content/drive/MyDrive/models/*/val_loss")
  kill_dir(x)

def get_model_choices():
  with open("/content/drive/MyDrive/functions/model_choices.txt","r") as f:
    data=f.readlines()
  ans=[]
  for i in data:
    try:
      ans.append(i.split()[0])
    except:
      break
  return ans

def write_model_section(section,train_dulation,val_dulation,num,train_white_cut):
  write_model_detail_path="/content/model/model_detail.txt"
  if os.path.exists(write_model_detail_path):
    os.remove(write_model_detail_path)
  f = open(write_model_detail_path, 'w')
  f.write("mode                     :"+section+"\n")
  f.write("train_dulation           :"+str(train_dulation)+"\n")
  f.write("val_dulation             :"+str(val_dulation)+"\n")
  f.write("seed_num                 :"+str(num)+"\n")
  f.write("train_white_cut          :"+str(train_white_cut)+"\n")
  f.close()

def get_labels():
  labels=[0,1,2,3,4,5,6,7,8,9]
  if os.path.exists("/content/based_image_data"):
    labels=glob.glob("/content/based_image_data/*")
    for i in range(len(labels)):
      labels[i]=labels[i].split("/")[-1]
  return labels

def start_make_dir(labels,all_img_path,all_label_path,train_index,val_index):
  #ここでtrain_dataとval_dataの分割を実施
  #また以前のものが残っている場合，ここで削除する
  #よって，最後の交差検証の場合，最後にデータが残存する
  for each_dir in labels:
    each_dir_path="/content/train_data/"+each_dir
    if os.path.exists(each_dir_path):
      shutil.rmtree(each_dir_path)
    os.mkdir(each_dir_path)
    each_val_dir_path="/content/val_data/"+each_dir
    if os.path.exists(each_val_dir_path):
      shutil.rmtree(each_val_dir_path)
    os.mkdir(each_val_dir_path)
  for k in train_index:
    shutil.copy(all_img_path[k],"/content/train_data/"+labels[all_label_path[k]])
  for k in val_index:
    shutil.copy(all_img_path[k],"/content/val_data/"+labels[all_label_path[k]])

def separate_img_data(amount,verifection):
  shutil.rmtree("/content/image_data",ignore_errors=True)
  z=glob.glob("/content/based_image_data/*")
  for i in z:
    if not os.path.exists("/content/image_data/"+i.split("/")[-1]):
      os.makedirs("/content/image_data/"+i.split("/")[-1])
    count=0
    z1=glob.glob(i+"/*")
    exclusion_data=[]
    if os.path.exists("/content/verifection_datas/"+i.split("/")[-1]) and verifection==True:
      f = open("/content/verifection_datas/"+i.split("/")[-1]+"/NO_1.txt", 'r')
      exclusion_data = f.readlines()
      f.close()
    for aa in range(len(exclusion_data)):
      exclusion_data[aa]=exclusion_data[aa].replace( '\n' , '' )
    for j in z1:
      if j in exclusion_data:
        continue
      shutil.copy(j,"/content/image_data/"+i.split("/")[-1])
      count+=1
      if count==amount:
        break


def plot_confusion_matrix(labels,fig,ax,i_count,cm, num,classes ,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    sns.heatmap(cm, ax=ax[i_count,num],annot=True,square=True,cmap='Blues',fmt="d",xticklabels=labels,yticklabels=labels)
    ax[i_count,num].set_xlabel("Predict_label")
    ax[i_count,num].set_ylabel("Correct_label")



def separate_train_val_test():
  X=[]
  Y=[]
  X_test=[]
  Y_test=[]

  char_data=glob.glob("/content/based_image_data/*")
  a=0
  class_num=-1
  print(char_data)
  for i in char_data:
    class_num=class_num+1
    data_files=glob.glob(i+"/*")
    data_files=sorted(data_files, key=lambda s: int(re.search(r'\d+', s).group()))
    print(data_files)
    each_x=[]
    each_y=[]
    each_test_x=[]
    each_test_y=[]
    for x in data_files:
      if len(each_test_x)<=19:
        each_test_x.append(x)
        each_test_y.append(class_num)
      elif len(each_x)<=119:
        each_x.append(x)
        each_y.append(class_num)
    X.extend(each_x)
    Y.extend(each_y)
    X_test.extend(each_test_x)
    Y_test.extend(each_test_y)

  return X,Y,X_test,Y_test

def train_test_copy_img(train_index,test_index,all_img_path,all_label_path):
  label=get_labels()
  shutil.rmtree("/content/train_data",ignore_errors=True)
  shutil.rmtree("/content/val_data",ignore_errors=True)  
  for i in label:
    os.makedirs("/content/train_data/"+i)
    os.makedirs("/content/val_data/"+i)
  for i in train_index:
    a=all_img_path[i].split("/")
    path="/content/train_data/"+a[-2]+"/"+a[-1]
    shutil.copyfile(all_img_path[i],path)
  for i in test_index:
    a=all_img_path[i].split("/")
    path="/content/val_data/"+a[-2]+"/"+a[-1]
    shutil.copyfile(all_img_path[i],path)

def make_test_dirs(files):
  label=get_labels()
  shutil.rmtree("/content/test_data",ignore_errors=True)
  for i in label:
    os.makedirs("/content/test_data/"+i)
  for i in files:
    a=i.split("/")
    path="/content/test_data/"+a[-2]+"/"+a[-1]
    shutil.copyfile(i,path)

def make_all_img_path_function():
  label_num=0
  all_img_path_datas=[]
  all_label_path_datas=[]
  first_img_dir_path="/content/image_data"
  img_top_dir=os.listdir(first_img_dir_path)
  for i in img_top_dir:
    count=1
    second_img_dir_path=os.path.join(first_img_dir_path,i)
    if os.path.isdir(second_img_dir_path):
      s=os.listdir(second_img_dir_path)
      for t in s:
        u=os.path.join(second_img_dir_path,t)
        if os.path.isfile(u):
          all_img_path_datas.append(u)
          all_label_path_datas.append(label_num)
          count=count+1
    label_num=label_num+1
  return all_img_path_datas ,all_label_path_datas

def reset_data():
  dir=glob.glob("/content/drive/MyDrive/sample/*")
  num_data=-1
  print("削除を行ってから行います")
  for i in dir:
    if os.path.isdir(i):
      num=len(glob.glob(i+"/*"))
      num_data=max(num,num_data)
  recursive_file_check("/content/drive/MyDrive/sample",num_data)
  shutil.rmtree("/content/model",ignore_errors=True)
  shutil.copytree("/content/drive/MyDrive/sample","/content/model")
  shutil.rmtree("/content/drive/MyDrive/sample",ignore_errors=True)
  return num_data

def train_models(args,num,labels):
  n_splits=6
  ss = ShuffleSplit(n_splits=6, test_size=0.25,random_state=0)
  all_acc=all_loss=all_val_acc=all_val_loss=[]
  last_acc=last_loss=last_val_acc=last_val_loss=[]
  i_count=0
  if args.test_separate==False:
    fig,ax=plt.subplots(n_splits,3,figsize=(25,40))
  else :
    fig,ax=plt.subplots(n_splits,4,figsize=(25,40))
  plt.subplots_adjust(wspace=0.4, hspace=0.6)
  
  all_img_path,all_label_path=make_all_img_path_function()
  print("all_img",len(all_img_path))
  data_time=datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
  labels=get_labels()
  
  #all_img_path,all_label_path,test_img_path,test_label_path=separate_train_val_test()
  for train_index,val_index in ss.split(all_img_path,all_label_path):
    if args.mode_add==True and os.path.exists("/content/drive/MyDrive/sample"):
      pass_num=reset_data()
    #shutil.rmtree("/content/drive/MyDrive/sample")
    print("----------------"+str(i_count+1)+"--------------------")
    train_index=sorted(train_index)
    val_index=sorted(val_index)

    if args.test_separate ==True:
      test_dir="/content/test_acc"
      shutil.rmtree(test_dir,ignore_errors=True)
      if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        label=get_labels()
        for i in label:
          os.makedirs(test_dir+"/"+i)
        for i in val_index:
          a=all_img_path[i].split("/")          
          path="/content/test_acc/"+a[-2]+"/"+a[-1]
          shutil.copyfile(all_img_path[i],path)
        train_index, val_index = train_test_split(train_index,random_state=42)
    
    
    train_test_copy_img(train_index,val_index,all_img_path,all_label_path)
    if os.path.exists("/content/test_acc"):
      print("test : ",len(glob.glob("/content/test_acc/*/*")))
      pass
    train_dulation=args.DA
    val_dulation=False

    L2_constrain=False
    

    train_white_cut=False
    val_white_cut=False

    if train_dulation==True:
      argumation_operation.train_dulation(i_count,train_white_cut)
    if val_dulation==True:
      argumation_operation.val_dulation(i_count,val_white_cut)
    if train_dulation ==False and val_dulation==False and i_count==0:
      print("水増しなし")
    black_white=False
    mode=args.color_mode
    if black_white==True:
      mode=1
      img_data_dir=["/content/train_data","/content/val_data"]
      for i in img_data_dir:
        for j in list_pictures(i):
          im=cv2.imread(j)
          im_gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
          cv2.imwrite(j, im_gray)
    
    size_data=0

    CNN_names=["CNN_NO1","CNN_NO1_GRAYSCALE","CNN_NO2","CNN_NO2_GRAYSCALE","SVM"]
    
    if args.arch in CNN_names:
      size_data=64
    else :
      size_data=224


    if args.arch!="SVM":
      X_train,y_train=func.make_arrays_num("/content/train_data",black_white,labels,mode,size_data)
      X_test,y_test=func.make_arrays_num("/content/val_data",black_white,labels,mode,size_data)
    else:
      X_train,y_train=func.make_arrays_num_SVM("/content/train_data",labels,mode,size_data)
      X_test,y_test=func.make_arrays_num_SVM("/content/val_data",labels,mode,size_data)
      if args.test_separate ==True:
        X_kensyo,y_kensyo=func.make_arrays_num("/content/test_acc",black_white,labels,mode,size_data)
    
    all_training_img_path=glob.glob("/content/train_data/*")
    validation_label=[]
    for each_y_test in y_test:
      num=1
      for each_each_y_test in each_y_test:
        if each_each_y_test==1:
          validation_label.append(num)
          break
        num=num+1 

    model_number=1
    parameter_optimizer=False
    section=func.Get_section(model_number,parameter_optimizer)


    if i_count==0:
      with open('/content/model/args.txt' , 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)
      write_model_section(section,train_dulation,val_dulation,num,train_white_cut)
      print("mode:",section)
      print("train_data",len(X_train))
      print("val_data",len(X_test))
    
    backup_dir = '/content/model'
    


    model=create_model.get_model(args,mode,section,X_train)
    if i_count==0:
      try:
        print(model.summary())
      except:
        pass
    if args.arch=="VGG_16_trans_SVM" or args.arch=="MobileNetV2_trans_SVM" or args.arch=="SVM" or args.arch=="InceptionV3_trans_SVM" or args.arch=="VGG_19_trans_SVM" or args.arch=="ResNet_trans_SVM":
      if args.arch=="VGG_16_trans_SVM" or args.arch=="MobileNetV2_trans_SVM" or args.arch=="InceptionV3_trans_SVM" or args.arch=="VGG_19_trans_SVM" or args.arch=="ResNet_trans_SVM":
        X_train_vgg16 = model.predict(X_train)
        X_test_vgg16 = model.predict(X_test)
        print(args.test_separate)
        if args.test_separate ==True:
          X_kensyo_VGG16=model.predict(X_kensyo)
          pass
        svm = SVC(kernel='linear', random_state=None)
      
        svm.fit(X_train_vgg16, np.argmax(y_train,axis = 1))
        pred_y=svm.predict(X_test_vgg16)
        pred_y_classes=pred_y
        if args.test_separate==True:
          pred_y_kensyo=svm.predict(X_kensyo_VGG16)

      elif args.arch=="SVM":
        print("SVM")
        svm = SVC(kernel='linear', random_state=None)
        svm.fit(X_train, np.argmax(y_train,axis = 1))
        pred_y=svm.predict(X_test)
        pred_y_classes=pred_y
      true_y= np.argmax(y_test,axis = 1) 
      f = open('/content/model/model_detail.txt', 'r')
      model_details = f.readlines()
      for i in range(len(model_details)):
        model_details[i]=model_details[i].rstrip('\n').split(":")[1]
      csv_dir_path="/content/model/csv_files"
      if not os.path.exists(csv_dir_path):
        os.makedirs(csv_dir_path)
  
      target_name=labels
      report = classification_report(true_y, pred_y_classes,target_names=target_name,digits=4)
      csv_path=csv_dir_path+"/No."+str(i_count+1)+".csv"
      func.classifaction_report_csv(report,csv_path)
      if args.test_separate==True:
        csv_dir_path="/content/model/csv_files_test"
        if not os.path.exists(csv_dir_path):
          os.makedirs(csv_dir_path)
        report = classification_report(true_y, pred_y_kensyo,target_names=target_name,digits=4)
        csv_path=csv_dir_path+"/No."+str(i_count+1)+".csv"
        func.classifaction_report_csv(report,csv_path)
      i_count=i_count+1
      continue
    

    if args.arch!="SVM":
      if not os.path.exists('/content/model/identification_model.json') :
        model_json_str = model.to_json()
        with open('/content/model/identification_model.json', 'w') as f:
          f.write(model_json_str)
      """
      # モデルの保存
      if not os.path.exists(backup_dir + '/identification_model.json'):
        model_json_str = model.to_json()
        with open(backup_dir + '/identification_model.json', 'w') as f:
          f.write(model_json_str)
      """
      BATCH_SIZE=args.batch_size
      EPOCHS=args.epochs
      # 重みデータのバックアップ
      #cb_cp = tf.keras.callbacks.ModelCheckpoint("/content/drive/MyDrive/functions/weights.hdf5", verbose=0, save_weights_only=True,save_best_only=True)
      cb_cp = tf.keras.callbacks.ModelCheckpoint(backup_dir + "/No."+str(i_count+1)+".weights.hdf5", verbose=0, save_weights_only=True,save_best_only=True)
      # TensorBoard用のデータ
      cb_tf = tf.keras.callbacks.TensorBoard(log_dir=backup_dir + '/tensorBoard', histogram_freq=0)
      early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')



    if args.arch=="SVM":
      model.fit(X_train, y_train)
      y_predicted = model.predict(X_test)
      print(y_predicted)
      print(type(y_predicted))
      print(y_predicted.shape)
      i_count=i_count+1
      continue
    callbacks_list=[cb_cp, cb_tf]
    if args.scheduler == 'CosineAnnealing':
        callbacks_list.append(CosineAnnealingScheduler(T_max=args.epochs, eta_max=args.lr, eta_min=args.min_lr, verbose=1))
    if args.early_stop==True:
      callbacks_list.append(early_stopping)




    if "MLP" in args.arch:
      history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,validation_data = (X_test, y_test), verbose = 1,  callbacks=callbacks_list)
    elif args.arch=="CNN_NO1" or args.arch=="CNN_NO2" or args.arch=="VGG_16_trans_MLP" or args.arch=="MobileNetV2_trans_MLP" or args.arch=="ResNet_trans_MLP" :
      history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,validation_data = (X_test, y_test), verbose = 1,  callbacks=callbacks_list)
    

    elif args.arch=="VGG_16_fine_tuning":
      history = model.fit(X_train,y_train,validation_data=(X_test,y_test), batch_size=BATCH_SIZE, epochs=EPOCHS,callbacks=callbacks_list, verbose = 1)
    try:
      shutil.copy((backup_dir + "/No."+str(i_count+1)+".weights.hdf5",  "/content/drive/MyDrive/parameters/No."+str(i_count+1)+".weights.hdf5"))
    except:
      print("コピーしっぱい")
      pass

    loss=history.history['loss']
    val_loss=history.history['val_loss']
    acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']

    func.records(i_count,history,labels)
    data_length=len(loss)-1
    all_loss.append(loss)
    last_loss.append(loss[data_length])
        
    all_val_loss.append(val_loss)
    last_val_loss.append(val_loss[data_length])
        
    all_acc.append(acc)
    last_acc.append(acc[data_length])

    all_val_acc.append(val_acc)
    last_val_acc.append(val_acc[data_length])
        
    ax[i_count,0].plot(acc)
    ax[i_count,0].plot(val_acc)
    ax[i_count,0].set_title('model accuracy')
    ax[i_count,0].set_xlabel('epoch')
    ax[i_count,0].set_ylabel('accuracy')
    ax[i_count,0].set_ylim([0.5, 1.0])
    ax[i_count,1].plot(loss)
    ax[i_count,1].plot(val_loss)
    ax[i_count,1].set_title('model loss')
    ax[i_count,1].set_ylabel('loss')
    ax[i_count,1].set_xlabel('epoch')
    ax[i_count,1].set_ylim([0, 2.0])
    
    
    del model
    
    model_pred = model_from_json(open(backup_dir + '/identification_model.json').read())
    model_pred.load_weights(backup_dir + "/No."+str(i_count+1)+".weights.hdf5")
    

    X_kensyo,y_kensyo=func.make_arrays_num("/content/val_data",black_white,labels,mode,size_data)
    pred_y = model_pred.predict(X_kensyo)

    pred_y_classes = np.argmax(pred_y,axis = 1) 
    true_y= np.argmax(y_kensyo,axis = 1) 
    
    confusion_mtx = confusion_matrix(true_y, pred_y_classes) 
    f = open('/content/model/model_detail.txt', 'r')
    model_details = f.readlines()
    for i in range(len(model_details)):
      model_details[i]=model_details[i].rstrip('\n').split(":")[1]
    csv_dir_path="/content/model/csv_files"
    if not os.path.exists(csv_dir_path):
      os.makedirs(csv_dir_path)
    #plot_confusion_matrix(labels,fig,ax,i_count,cm, classes,num ,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    #plot_confusion_matrix(labels,fig,ax,i_count,cm, classes,num ,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plot_confusion_matrix(labels,fig,ax,i_count,confusion_mtx,2,classes = range(3))
    target_name=labels
    report = classification_report(true_y, pred_y_classes,target_names=target_name,digits=4)
    csv_path=csv_dir_path+"/No."+str(i_count+1)+".csv"
    func.classifaction_report_csv(report,csv_path)


    if os.path.exists("/content/test_acc"):
      X_kensyo,y_kensyo=func.make_arrays_num("/content/test_acc",black_white,labels,mode,size_data)
      pred_y = model_pred.predict(X_kensyo)

      pred_y_classes = np.argmax(pred_y,axis = 1) 
      true_y= np.argmax(y_kensyo,axis = 1) 

      confusion_mtx = confusion_matrix(true_y, pred_y_classes) 
      f = open('/content/model/model_detail.txt', 'r')
      model_details = f.readlines()

      for i in range(len(model_details)):
        model_details[i]=model_details[i].rstrip('\n').split(":")[1]
      csv_dir_path="/content/model/csv_files_test"
      if not os.path.exists(csv_dir_path):
        os.makedirs(csv_dir_path)
      #plot_confusion_matrix(labels,fig,ax,i_count,cm, classes,num ,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
      plot_confusion_matrix(labels,fig,ax,i_count,confusion_mtx,3,classes = range(3))
      target_name=labels
      report = classification_report(true_y, pred_y_classes,target_names=target_name,digits=4)
      csv_path=csv_dir_path+"/No."+str(i_count+1)+".csv"
      func.classifaction_report_csv(report,csv_path)


    if os.path.exists("/content/model/img.png"):
      os.remove("/content/model/img.png")
    fig.savefig("/content/model/img.png")
    i_count=i_count+1

def model_train(args,num):
  backup_dir = '/content/model'
  txt_file=""
  if args.persons_selected==False:
    ascending_class_file_path="/content/drive/MyDrive/functions/written_classes_on_ascending.txt"
    if not os.path.exists(ascending_class_file_path):
      print("ERROR:昇順降順が記入されたtxtデータが存在しません．")
      sys.exit()
    else:
      txt_file=ascending_class_file_path
  if args.persons_selected==True:
    target_class="/content/target_class.txt"
    if not  os.path.exists(target_class):
      print("ERROR:対象とする人物classが存在してません")
      print("target_class.txtにclassを記入してください")
      with open(target_class,"w") as f:
        pass
      sys.exit()     
    else:
      txt_file=target_class
  f = open(txt_file, 'r')
  datalist = f.readlines()
  for i in range(len(datalist)):
    datalist[i]=datalist[i].split("\n")[0]
  if args.add_ordered=="descending_order":
    datalist.reverse()
  datalist=datalist[:num]
  
  labels=datalist

  MAKE_DIR.operate_move_data(args,datalist)


  if args.open_world==True:
    classified_others.hold_data()
  if args.img_num!=140:
    dir_data=glob.glob("/content/based_image_data/*")
    for each_dir_data in dir_data:
      files_path=glob.glob(each_dir_data+"/*")
      natsorted(files_path)

      kill_data=files_path[120:]
      for j in kill_data:
        try:
          os.remove(j)
        except:
          pass
  verifection=False
  separate_img_data(120,verifection)
  shutil.rmtree("/content/model",ignore_errors=True)
  shutil.rmtree("/content/train_data",ignore_errors=True)
  shutil.rmtree("/content/val_data",ignore_errors=True)
  os.makedirs(backup_dir)
  with open(backup_dir + '/labels.txt', 'w') as f:
    for d in labels:
      if labels[-1]!=d:
        f.write("%s\n" % d)
      else:
        f.write("%s" % d)
  train_models(args,10,labels)
  Preds_Data_Func.pred_parameters(labels)
  func.end_functions()