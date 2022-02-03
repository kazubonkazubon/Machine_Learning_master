import module_1 as func
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.preprocessing.image import array_to_img, img_to_array, load_img
import pandas as pd
import shutil
import os
import cv2
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import model_from_json
import glob
import pytz
import openpyxl
import subprocess
import tensorflow.keras.backend as K


class L2ConstrainLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(L2ConstrainLayer, self).__init__()
        self.alpha = tf.Variable(30.)

    def call(self, inputs):
        #about l2_normalize https://www.tensorflow.org/api_docs/python/tf/keras/backend/l2_normalize?hl=ja
        return K.l2_normalize(inputs, axis=1) * self.alpha

def get_filter_num():
  f = open('/content/input_size.txt', 'r')
  num=int(f.read())
  return num

def pred(img_path,labels,df,index_numbers,j,model_pred,all_data,all_data_labels,mode):
  """
  input
    img_path          :分類を行う画像のfull path or relative path
    labels            :今回学習時に使用した分類クラス
    df                : 結果をまとめるためのdata frame
    index_numbers     :
    j                 :
    model_pred        :
    all_data          :
    all_data_labels   :
    mode              :RGBだと3 Gray_Scaleだと1で計算を行う
  
  return 
    
  """
  backup_dir = '/content/model'
  X = []
  filter_num=get_filter_num()
  if mode==3:
    img = img_to_array(load_img(img_path, target_size=(filter_num,filter_num),color_mode = 'rgb'))
  if mode==1:
    img = img_to_array(load_img(img_path, target_size=(filter_num,filter_num),color_mode = 'grayscale'))
  X.append(img)
  X = np.asarray(X)
  X = X.astype('float32')
  X = X / 255.0
  preds = model_pred.predict(X,verbose=0)
  pred_label = ""
  label_num = 0
  all_data.append(str(max(preds[0])))
  all_data_labels.append(labels[np.argmax(preds)])
  np.set_printoptions(suppress=True)
  n=img_path.split("/")

  k=[]
  k.append(n[-2]+"/"+n[-1])
  first_data=0
  second_data=0
  for each_data in preds[0]:
    each_data=round(float(each_data),5)
    if first_data==0:
      first_data=each_data
    elif first_data<each_data:
      second_data=first_data
      first_data=each_data
    elif second_data<each_data:
      second_data=each_data
    k.append(each_data)
  k.append(labels[np.argmax(preds)])
  k.append(n[-2])
  if n[-2]==labels[np.argmax(preds)]:
    k.append("True")
    k.append(1)
  else:
    k.append("False")
    k.append(0)
  k.append(first_data-second_data)
  df.loc[str(index_numbers)] = k
  return all_data,all_data_labels

def pred_parameters(labels):
  f = open('/content/mode_data.txt', 'r',)
  mode = f.read()
  mode=int(mode)
  f.close()
  backup_dir = '/content/model'
  sys.path.append("/content/drive/MyDrive/functions/")
  index_datas=["data_path"]+labels+["Pred_class","Real_class","correct_wrong","True_num","residual eroor"]
  directory="/content/val_data"
  # get images which need pred data 
  image_path= func.list_pictures(directory)
  if os.path.exists("/content/pred_data"):
    new_image_path=func.list_pictures("/content/pred_data")
    image_path.extend(new_image_path)
  model_list=glob.glob("/content/model/*.hdf5")
  model_number=len(model_list)
  f = open('/content/model/model_detail.txt', 'r')
  model_details = f.readlines()
  for i in range(len(model_details)):
    model_details[i]=model_details[i].rstrip('\n').split(":")[1]
  if model_details[1]=="True" and model_details[2] =="True":
    excel_sub_dir_name="mode "+model_details[0]+" with all dulation seed num "+model_details[-1]
  if model_details[1]=="True" and model_details[2] =="False":
    excel_sub_dir_name="mode "+model_details[0]+" with train dulation seed num "+model_details[-1]
  if model_details[1]=="False" and model_details[2] =="True":
    excel_sub_dir_name="mode "+model_details[0]+" with val dulation seed num "+model_details[-1]
  if model_details[1]=="False" and model_details[2] =="False":
    excel_sub_dir_name="mode "+model_details[0]+" with no dulation seed num "+model_details[-1]
  excel_sub_dir="/content/model/predict/"+excel_sub_dir_name
  shutil.rmtree(excel_sub_dir,ignore_errors=True)
  os.makedirs(excel_sub_dir)
  for j in range(len(model_list)):
    df = pd.DataFrame(index=[], columns=index_datas)
    all_data=[]
    all_data_labels=[]
    index_numbers=1
    model_pred=model_from_json(open(backup_dir + '/identification_model.json').read())
    model_pred.load_weights("/content/model/No."+str(j+1)+".weights.hdf5")
    #imgge_path : type(List) pred all data in this section
    for i in image_path:
      all_data,all_data_labels=pred(i,labels,df,index_numbers,j+1,model_pred,all_data,all_data_labels,mode)
      index_numbers=index_numbers+1
    excel_path="/content/model/predict/"+excel_sub_dir_name+"/predict_NO."+str(j+1)+".xlsx"
    df_true=df.query('correct_wrong == "True"')
    df_false=df.query('correct_wrong == "False"')
    with pd.ExcelWriter(excel_path) as writer:
      df.to_excel(writer, sheet_name='predict')
      df_true.to_excel(writer, sheet_name='TRUE')
      df_false.to_excel(writer, sheet_name='FALSE')